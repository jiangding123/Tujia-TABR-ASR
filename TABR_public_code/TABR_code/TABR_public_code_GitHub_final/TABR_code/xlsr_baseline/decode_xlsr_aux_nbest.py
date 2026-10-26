#!/usr/bin/env python3
"""
Generate main-CTC N-best and exact sequence CTC scores.

For TABR, --main_ctc_only avoids loading or evaluating FA auxiliary heads;
only the separately trained Base CTC model supplies the additional score.
The original auxiliary scoring mode remains available for other experiments.

This script DOES NOT choose alpha/beta weights. It creates a reusable cache so
weight search on Dev is CPU-only and does not repeatedly run XLS-R-300M.
"""
import argparse
import json
import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
)

from auxiliary_projection import (
    build_main_to_aux_projection,
)
from common import (
    SPECIAL_TOKENS,
    load_audio,
    load_symbol_table,
    read_jsonl,
    resolve_wav_path,
)
from ctc_rescore_utils import (
    ctc_prefix_beam_search,
    ctc_sequence_log_scores,
)


class DecodeDataset(Dataset):
    def __init__(
        self,
        list_path: str,
        sampling_rate: int,
        root: str = ".",
    ):
        self.rows = read_jsonl(list_path)
        self.sampling_rate = sampling_rate
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        waveform = load_audio(
            resolve_wav_path(
                row["wav"], self.root
            ),
            self.sampling_rate,
        )

        return {
            "key": row["key"],
            "waveform": waveform,
            "ref_tokens": row["txt"].split(),
        }


class DecodeCollator:
    def __init__(
        self,
        feature_extractor,
        sampling_rate: int,
    ):
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __call__(self, items: List[dict]):
        waves = [
            x["waveform"].numpy()
            for x in items
        ]

        batch = self.feature_extractor(
            waves,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch["keys"] = [
            x["key"] for x in items
        ]
        batch["ref_tokens"] = [
            x["ref_tokens"] for x in items
        ]

        return batch


def load_linear_head(
    state,
    hidden_size: int,
    device,
):
    if state is None:
        return None

    if (
        "weight" not in state
        or "bias" not in state
    ):
        raise ValueError(
            "auxiliary head state_dict must contain "
            "weight and bias"
        )

    out_features, in_features = (
        state["weight"].shape
    )

    if int(in_features) != int(hidden_size):
        raise ValueError(
            "auxiliary head hidden-size mismatch: "
            f"checkpoint={in_features}, "
            f"main model={hidden_size}"
        )

    head = nn.Linear(
        int(in_features),
        int(out_features),
    )
    head.load_state_dict(
        state, strict=True
    )

    return head.to(device).eval()


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model_dir", required=True
    )
    p.add_argument(
        "--data_list", required=True
    )
    p.add_argument(
        "--dict_path", required=True
    )
    p.add_argument(
        "--output_jsonl", required=True
    )
    p.add_argument(
        "--root", default="."
    )
    p.add_argument(
        "--task",
        choices=["fnx", "fax", "fjx"],
        required=True,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )
    p.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
    )
    p.add_argument(
        "--blank_id",
        type=int,
        default=0,
    )
    p.add_argument(
        "--beam_size",
        type=int,
        default=10,
    )
    p.add_argument(
        "--token_prune",
        type=int,
        default=20,
        help=(
            "Per-frame top-k valid labels used only "
            "inside prefix beam candidate generation. "
            "0 disables this pruning."
        ),
    )
    p.add_argument(
        "--fp16",
        action="store_true",
    )
    p.add_argument(
        "--main_ctc_only", "--main-ctc-only",
        action="store_true",
        help=(
            "TABR inference mode: decode and score using FA main CTC only. "
            "Do not load auxiliary_heads.pt or compute auxiliary scores. "
            "Requires --task fax; tone_score/notone_score will be null "
            "until an external Base model attaches its scores."
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "❌ decoding is configured for CUDA"
        )

    if args.main_ctc_only and args.task != "fax":
        raise SystemExit(
            "--main_ctc_only is intended for FA TABR decoding: "
            "use --task fax"
        )

    device = torch.device("cuda")
    model_dir = Path(args.model_dir)

    _, id2token = load_symbol_table(
        args.dict_path
    )

    main_vocab_size = max(id2token) + 1

    if main_vocab_size != 871:
        raise SystemExit(
            "❌ expected main vocab size 871, "
            f"got {main_vocab_size}"
        )

    if (
        id2token.get(args.blank_id)
        != "<blank>"
    ):
        raise SystemExit(
            f"❌ expected blank id "
            f"{args.blank_id} = <blank>"
        )

    model = (
        Wav2Vec2ForCTC
        .from_pretrained(model_dir)
        .to(device)
        .eval()
    )

    if int(model.config.vocab_size) != 871:
        raise SystemExit(
            "❌ main checkpoint vocab mismatch: "
            f"{model.config.vocab_size}"
        )

    feature_extractor = (
        Wav2Vec2FeatureExtractor
        .from_pretrained(model_dir)
    )

    # The main FA encoder/CTC head is independent of the auxiliary heads.
    # Keeping this branch separate guarantees the TABR path never opens
    # auxiliary_heads.pt, instantiates auxiliary layers, or projects labels.
    aux = None
    notone_head = None
    tone_head = None
    main_to_notone = None
    main_to_tone = None

    if not args.main_ctc_only:
        aux_path = model_dir / "auxiliary_heads.pt"
        if not aux_path.is_file():
            raise SystemExit(
                f"missing auxiliary checkpoint: {aux_path}"
            )
        aux = torch.load(aux_path, map_location="cpu")

        required = {
            "use_notone", "use_tone", "notone_head", "tone_head",
            "notone_token2id", "tone_token2id", "tone_regex",
            "unmarked_tone_token",
        }
        missing = required - set(aux)
        if missing:
            raise SystemExit(
                "auxiliary_heads.pt missing keys: "
                f"{sorted(missing)}"
            )

        if args.task == "fnx" and not bool(aux["use_notone"]):
            raise SystemExit("FN-X requires a saved no-tone head")
        if args.task == "fax" and not bool(aux["use_tone"]):
            raise SystemExit("FA-X requires a saved tone head")
        if args.task == "fjx" and (
            not bool(aux["use_notone"]) or not bool(aux["use_tone"])
        ):
            raise SystemExit("FJ-X requires both auxiliary heads")

        hidden_size = int(model.config.hidden_size)
        notone_head = load_linear_head(
            aux["notone_head"], hidden_size, device
        )
        tone_head = load_linear_head(
            aux["tone_head"], hidden_size, device
        )
        (
            main_to_notone, main_to_tone, _split_map
        ) = build_main_to_aux_projection(
            id2token=id2token,
            notone_token2id=aux["notone_token2id"],
            tone_token2id=aux["tone_token2id"],
            tone_regex=str(aux["tone_regex"]),
            unmarked_tone_token=str(aux["unmarked_tone_token"]),
        )

    # Do not allow unused special symbols to appear
    # as ASR output labels.
    allowed_main_ids = [
        idx
        for idx, token
        in sorted(id2token.items())
        if (
            idx == args.blank_id
            or token not in SPECIAL_TOKENS
        )
    ]

    dataset = DecodeDataset(
        args.data_list,
        args.sampling_rate,
        args.root,
    )

    collator = DecodeCollator(
        feature_extractor,
        args.sampling_rate,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    output_path = Path(
        args.output_jsonl
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    meta_path = (
        output_path
        .with_suffix(".meta.json")
    )

    print("=" * 76)
    print(
        "XLS-R auxiliary CTC N-best scoring"
    )
    print("task:", args.task.upper())
    print("model:", model_dir)
    print("data:", args.data_list)
    print("utterances:", len(dataset))
    print("beam size:", args.beam_size)
    print(
        "token prune:", args.token_prune
    )
    print("main vocab:", 871)
    if args.main_ctc_only:
        print("decoding mode: FA MAIN CTC ONLY")
        print("FA auxiliary checkpoint: not loaded")
        print("FA auxiliary logits/scores: not computed")
    else:
        print("decoding mode: main + auxiliary CTC scores")
        print("no-tone vocab:", len(aux["notone_token2id"]))
        print("tone vocab:", len(aux["tone_token2id"]))
        print("saved TRAIN lambda_notone:", aux.get("lambda_notone"))
        print("saved TRAIN lambda_tone:", aux.get("lambda_tone"))
    print("Decoding alpha/beta are selected separately on Dev.")
    print("=" * 76)

    rows_written = 0

    with output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                loader, 1
            ):
                keys = batch.pop("keys")
                refs = batch.pop(
                    "ref_tokens"
                )

                input_values = batch[
                    "input_values"
                ].to(
                    device,
                    non_blocking=True,
                )

                attention_mask = batch[
                    "attention_mask"
                ].to(
                    device,
                    non_blocking=True,
                )

                with torch.cuda.amp.autocast(
                    enabled=bool(args.fp16)
                ):
                    enc = model.wav2vec2(
                        input_values=input_values,
                        attention_mask=attention_mask,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )

                    hidden = model.dropout(
                        enc.last_hidden_state
                    )

                    main_logits = (
                        model.lm_head(hidden)
                    )

                    notone_logits = (
                        notone_head(hidden)
                        if notone_head is not None
                        else None
                    )

                    tone_logits = (
                        tone_head(hidden)
                        if tone_head is not None
                        else None
                    )

                input_lengths = (
                    attention_mask.sum(-1)
                )

                output_lengths = (
                    model
                    ._get_feat_extract_output_lengths(
                        input_lengths
                    )
                    .to(torch.long)
                )

                for i, key in enumerate(keys):
                    time_steps = int(
                        output_lengths[i].item()
                    )

                    main_lp = F.log_softmax(
                        main_logits[
                            i, :time_steps
                        ].float(),
                        dim=-1,
                    )

                    beam = (
                        ctc_prefix_beam_search(
                            main_lp
                            .detach()
                            .cpu(),
                            beam_size=(
                                args.beam_size
                            ),
                            blank_id=(
                                args.blank_id
                            ),
                            allowed_token_ids=(
                                allowed_main_ids
                            ),
                            token_prune=(
                                args.token_prune
                            ),
                        )
                    )

                    candidate_ids = [
                        list(prefix)
                        for prefix, _
                        in beam
                    ]

                    # Exact main CTC sequence score for
                    # every retained beam candidate.
                    main_scores = (
                        ctc_sequence_log_scores(
                            main_lp,
                            candidate_ids,
                            blank_id=(
                                args.blank_id
                            ),
                        )
                    )

                    notone_scores = [
                        None
                    ] * len(candidate_ids)

                    tone_scores = [
                        None
                    ] * len(candidate_ids)

                    if (
                        notone_logits
                        is not None
                    ):
                        nt_targets = [
                            [
                                main_to_notone[
                                    token_id
                                ]
                                for token_id
                                in seq
                            ]
                            for seq
                            in candidate_ids
                        ]

                        nt_lp = F.log_softmax(
                            notone_logits[
                                i, :time_steps
                            ].float(),
                            dim=-1,
                        )

                        notone_scores = (
                            ctc_sequence_log_scores(
                                nt_lp,
                                nt_targets,
                                blank_id=0,
                            )
                        )

                    if tone_logits is not None:
                        tone_targets = [
                            [
                                main_to_tone[
                                    token_id
                                ]
                                for token_id
                                in seq
                            ]
                            for seq
                            in candidate_ids
                        ]

                        tone_lp = F.log_softmax(
                            tone_logits[
                                i, :time_steps
                            ].float(),
                            dim=-1,
                        )

                        tone_scores = (
                            ctc_sequence_log_scores(
                                tone_lp,
                                tone_targets,
                                blank_id=0,
                            )
                        )

                    candidates = []

                    for rank, seq in enumerate(
                        candidate_ids
                    ):
                        tokens = [
                            id2token[x]
                            for x in seq
                        ]

                        candidates.append({
                            "rank": rank,
                            "tokens": tokens,
                            "main_score": (
                                finite_or_none(
                                    main_scores[
                                        rank
                                    ]
                                )
                            ),
                            "notone_score": (
                                finite_or_none(
                                    notone_scores[
                                        rank
                                    ]
                                )
                                if (
                                    notone_scores[
                                        rank
                                    ]
                                    is not None
                                )
                                else None
                            ),
                            "tone_score": (
                                finite_or_none(
                                    tone_scores[
                                        rank
                                    ]
                                )
                                if (
                                    tone_scores[
                                        rank
                                    ]
                                    is not None
                                )
                                else None
                            ),
                        })

                    record = {
                        "key": key,
                        "ref_tokens": refs[i],
                        "candidates": (
                            candidates
                        ),
                    }

                    fout.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    rows_written += 1

                if (
                    batch_idx % 10 == 0
                    or batch_idx
                    == len(loader)
                ):
                    print(
                        "scored batches: "
                        f"{batch_idx}/"
                        f"{len(loader)} "
                        "utterances="
                        f"{rows_written}",
                        flush=True,
                    )

    meta = {
        "task": args.task,
        "model_dir": str(model_dir),
        "data_list": args.data_list,
        "dict_path": args.dict_path,
        "utterances": rows_written,
        "beam_size": args.beam_size,
        "token_prune": args.token_prune,
        "score_definition": (
            "exact_ctc_log_probability"
        ),
        "candidate_generator": (
            "main_ctc_prefix_beam_search"
        ),
        "decoding_mode": (
            "main_ctc_only" if args.main_ctc_only else "main_plus_aux"
        ),
        "auxiliary_checkpoint_loaded": not args.main_ctc_only,
        "auxiliary_scoring_enabled": not args.main_ctc_only,
        "saved_training_lambda_notone": (
            aux.get("lambda_notone") if aux is not None else None
        ),
        "saved_training_lambda_tone": (
            aux.get("lambda_tone") if aux is not None else None
        ),
        "use_notone": bool(aux["use_notone"]) if aux is not None else False,
        "use_tone": bool(aux["use_tone"]) if aux is not None else False,
    }

    meta_path.write_text(
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "✅ N-best score cache:",
        output_path,
    )
    print("metadata:", meta_path)


if __name__ == "__main__":
    main()
