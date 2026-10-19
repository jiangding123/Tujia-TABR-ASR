#!/usr/bin/env python3
"""
Attach exact CTC scores from a standalone NO-TONE MAIN XLS-R model to an
existing FA-generated full-tone N-best cache.

Output candidate fields:
  main_score   = FA main exact CTC score
  notone_score = standalone no-tone MAIN exact CTC score
  tone_score   = preserved if already present in the FA cache

Intended final score:
  S(y) = S_main_FA(y) + alpha * S_notone_MAIN(base(y))
"""
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from common import load_audio, load_symbol_table, read_jsonl, resolve_wav_path
from ctc_rescore_utils import ctc_sequence_log_scores


SPECIAL_TOKENS = {
    "<blank>", "<unk>", "<sos/eos>", "<sos>", "<eos>", "<pad>"
}


class ScoreDataset(Dataset):
    def __init__(self, list_path: str, sampling_rate: int, root: str = "."):
        self.rows = read_jsonl(list_path)
        self.sampling_rate = int(sampling_rate)
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        waveform = load_audio(
            resolve_wav_path(row["wav"], self.root),
            self.sampling_rate,
        )
        return {
            "key": row["key"],
            "waveform": waveform,
            "ref_tokens": row["txt"].split(),
        }


class ScoreCollator:
    def __init__(self, feature_extractor, sampling_rate: int):
        self.feature_extractor = feature_extractor
        self.sampling_rate = int(sampling_rate)

    def __call__(self, items: List[dict]):
        waves = [x["waveform"].numpy() for x in items]
        batch = self.feature_extractor(
            waves,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        batch["keys"] = [x["key"] for x in items]
        batch["ref_tokens"] = [x["ref_tokens"] for x in items]
        return batch


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def read_base_cache(path: str):
    rows = []
    key2row: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key")
            if not key:
                raise ValueError(f"{path}:{lineno}: missing key")
            if key in key2row:
                raise ValueError(f"{path}:{lineno}: duplicate key {key}")
            candidates = row.get("candidates")
            if not candidates:
                raise ValueError(f"{path}:{lineno}: no candidates")
            for candidate in candidates:
                if "tokens" not in candidate:
                    raise ValueError(f"{path}:{lineno}: candidate missing tokens")
                if "main_score" not in candidate:
                    raise ValueError(
                        f"{path}:{lineno}: candidate missing main_score"
                    )
            rows.append(row)
            key2row[key] = row
    if not rows:
        raise ValueError(f"empty base cache: {path}")
    return rows, key2row


def split_base(token: str, tone_pattern: re.Pattern) -> str:
    if token == "<unk>":
        return token
    m = tone_pattern.fullmatch(token)
    if m is None:
        return token
    base = m.group(1)
    tone = m.group(2)
    if not base or not tone:
        raise ValueError(f"invalid full-tone token split: {token!r}")
    return base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_cache", required=True)
    p.add_argument("--notone_model_dir", required=True)
    p.add_argument("--notone_dict_path", required=True)
    p.add_argument("--data_list", required=True)
    p.add_argument("--main_dict_path", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--sampling_rate", type=int, default=16000)
    p.add_argument("--blank_id", type=int, default=0)
    p.add_argument("--tone_regex", default=r"^(.*?)([0-9]+)$")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("❌ standalone no-tone scoring is configured for CUDA")

    device = torch.device("cuda")
    base_rows, base_key2row = read_base_cache(args.base_cache)

    _, full_id2token = load_symbol_table(args.main_dict_path)
    if max(full_id2token) + 1 != 871:
        raise SystemExit(
            "❌ expected full-tone main vocabulary size 871, "
            f"got {max(full_id2token) + 1}"
        )
    if full_id2token.get(args.blank_id) != "<blank>":
        raise SystemExit(
            f"❌ expected full-tone blank id {args.blank_id}=<blank>"
        )

    notone_token2id, notone_id2token = load_symbol_table(args.notone_dict_path)
    notone_vocab_size = max(notone_id2token) + 1
    if notone_id2token.get(args.blank_id) != "<blank>":
        raise SystemExit(
            f"❌ expected no-tone blank id {args.blank_id}=<blank>"
        )

    model_dir = Path(args.notone_model_dir)
    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"❌ missing standalone no-tone model: {model_dir}")

    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device).eval()
    if int(model.config.vocab_size) != int(notone_vocab_size):
        raise SystemExit(
            "❌ standalone no-tone checkpoint/dictionary vocab mismatch: "
            f"model={model.config.vocab_size}, dict={notone_vocab_size}"
        )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    if int(feature_extractor.sampling_rate) != int(args.sampling_rate):
        raise SystemExit(
            "❌ sampling-rate mismatch: "
            f"extractor={feature_extractor.sampling_rate}, "
            f"requested={args.sampling_rate}"
        )

    tone_pattern = re.compile(args.tone_regex)

    full_to_notone_id = {}
    for idx in sorted(full_id2token):
        token = full_id2token[idx]
        if token in SPECIAL_TOKENS:
            continue
        base = split_base(token, tone_pattern)
        if base not in notone_token2id:
            raise SystemExit(
                "❌ no-tone dictionary is incompatible with the full-tone "
                f"dictionary: {token!r} -> {base!r}, but {base!r} is missing "
                f"from {args.notone_dict_path}"
            )
        full_to_notone_id[token] = int(notone_token2id[base])

    dataset = ScoreDataset(args.data_list, args.sampling_rate, args.root)
    if len(dataset) != len(base_rows):
        raise SystemExit(
            "❌ data/cache utterance count mismatch: "
            f"data={len(dataset)}, cache={len(base_rows)}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ScoreCollator(feature_extractor, args.sampling_rate),
        persistent_workers=(args.num_workers > 0),
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(".meta.json")

    print("=" * 76)
    print("FA + standalone NO-TONE MAIN exact CTC scoring")
    print("base candidate/main scorer: FA main CTC")
    print("external no-tone scorer: standalone no-tone MAIN CTC")
    print("base cache:", args.base_cache)
    print("no-tone model:", model_dir)
    print("no-tone dict:", args.notone_dict_path)
    print("no-tone vocab:", notone_vocab_size)
    print("data:", args.data_list)
    print("utterances:", len(dataset))
    print("intended final score:")
    print("  S_main_FA + alpha * S_notone_MAIN")
    print("=" * 76)

    seen = set()
    output_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            keys = batch.pop("keys")
            refs = batch.pop("ref_tokens")
            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=True
            )

            with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                enc = model.wav2vec2(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden = model.dropout(enc.last_hidden_state)
                notone_logits = model.lm_head(hidden)

            input_lengths = attention_mask.sum(-1)
            output_lengths = model._get_feat_extract_output_lengths(
                input_lengths
            ).to(torch.long)

            for i, key in enumerate(keys):
                if key not in base_key2row:
                    raise SystemExit(f"❌ key missing from FA cache: {key}")

                base_row = base_key2row[key]
                if list(base_row["ref_tokens"]) != list(refs[i]):
                    raise SystemExit(f"❌ reference mismatch for key {key}")

                targets = []
                for candidate in base_row["candidates"]:
                    seq = []
                    for token in candidate["tokens"]:
                        if token not in full_to_notone_id:
                            raise SystemExit(
                                f"❌ candidate token {token!r} in {key} cannot "
                                "be projected into the standalone no-tone vocab"
                            )
                        seq.append(full_to_notone_id[token])
                    targets.append(seq)

                time_steps = int(output_lengths[i].item())
                nt_lp = F.log_softmax(
                    notone_logits[i, :time_steps].float(), dim=-1
                )
                nt_scores = ctc_sequence_log_scores(
                    nt_lp, targets, blank_id=args.blank_id
                )

                merged_candidates = []
                for rank, candidate in enumerate(base_row["candidates"]):
                    merged = dict(candidate)
                    merged["notone_score"] = finite_or_none(nt_scores[rank])
                    merged["score_sources"] = {
                        "main": "FA main CTC",
                        "tone": (
                            "FA tone CTC (preserved if present; beta=0 here)"
                            if candidate.get("tone_score") is not None
                            else "not computed (FA main-CTC-only decoding)"
                        ),
                        "notone": "standalone no-tone MAIN CTC",
                    }
                    merged_candidates.append(merged)

                merged_row = dict(base_row)
                merged_row["candidates"] = merged_candidates
                merged_row["score_sources"] = {
                    "candidate_generator": "FA main CTC",
                    "main_score": "FA main CTC",
                    "tone_score": (
                        "FA tone CTC if present; unused by TABR"
                        if any(c.get("tone_score") is not None for c in base_row["candidates"])
                        else "not computed (FA main-CTC-only decoding)"
                    ),
                    "notone_score": "standalone no-tone MAIN CTC",
                }
                output_rows.append(merged_row)
                seen.add(key)

            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(
                    f"standalone no-tone score batches: "
                    f"{batch_idx}/{len(loader)} utterances={len(seen)}",
                    flush=True,
                )

    missing_keys = set(base_key2row) - seen
    extra_keys = seen - set(base_key2row)
    if missing_keys or extra_keys:
        raise SystemExit(
            "❌ key mismatch after standalone no-tone scoring: "
            f"missing={len(missing_keys)} extra={len(extra_keys)}"
        )

    out_map = {row["key"]: row for row in output_rows}
    with output_path.open("w", encoding="utf-8") as f:
        for base_row in base_rows:
            f.write(
                json.dumps(out_map[base_row["key"]], ensure_ascii=False) + "\n"
            )

    meta = {
        "task": "fa_plus_standalone_notone_main",
        "score_definition": "S_main_FA + alpha*S_notone_MAIN",
        "candidate_generator": "FA main CTC prefix beam",
        "base_cache": args.base_cache,
        "notone_model_dir": str(model_dir),
        "notone_dict_path": args.notone_dict_path,
        "notone_vocab_size": notone_vocab_size,
        "data_list": args.data_list,
        "main_dict_path": args.main_dict_path,
        "utterances": len(base_rows),
        "score_sources": {
            "main_score": "FA main CTC",
            "notone_score": "standalone no-tone MAIN CTC",
            "tone_score": "not computed in main-only mode; otherwise preserved if present",
        },
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ merged FA + standalone no-tone cache:", output_path)
    print("metadata:", meta_path)


if __name__ == "__main__":
    main()
