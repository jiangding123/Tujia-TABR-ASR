#!/usr/bin/env python3
"""
XLS-R-300M multi-task CTC fine-tuning for Tujia.

Main task:
    full tone-marked Tujia syllable CTC (same 871-way task as baseline)

Auxiliary tasks:
    no-tone syllable CTC
    tone CTC

The auxiliary target sequences are derived from the main transcript tokens.
For example:
    long3 sang1 nie2
becomes:
    no-tone: long sang nie
    tone:    3    1    2

This script intentionally does NOT depend on notone_frame/tone_frame/aux_segment.
Those fields are alignment-oriented supervision used by other experiments.
Here every utterance can participate in the sequence-level auxiliary CTC tasks.

The best checkpoint is still selected only by MAIN-task Dev CER.
The saved best_model/ is a normal Hugging Face Wav2Vec2ForCTC directory,
so the existing decode_xlsr_tujia.py can decode it without modification.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    get_linear_schedule_with_warmup,
)

from common import (
    SPECIAL_TOKENS,
    collapse_ctc,
    compute_ser_cer,
    format_cer,
    format_ser,
    load_audio,
    load_symbol_table,
    read_jsonl,
    resolve_wav_path,
    set_seed,
    text_to_ids,
    write_text,
)


AUX_SPECIALS = {
    "<blank>": 0,
    "<unk>": 1,
    "<sos/eos>": 2,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--exp_dir", default=None)
    p.add_argument("--lambda_notone", type=float, default=None)
    p.add_argument("--lambda_tone", type=float, default=None)
    p.add_argument(
        "--check_only",
        action="store_true",
        help="Validate config/transcripts/auxiliary token mapping without loading XLS-R.",
    )
    return p.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(conf, args):
    mapping = {
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "exp_dir": args.exp_dir,
        "lambda_notone": args.lambda_notone,
        "lambda_tone": args.lambda_tone,
    }
    for key, value in mapping.items():
        if value is not None:
            conf[key] = value


def validate_aux_config(conf):
    use_notone = bool(conf.get("use_notone", False))
    use_tone = bool(conf.get("use_tone", False))
    lambda_notone = float(conf.get("lambda_notone", 0.0))
    lambda_tone = float(conf.get("lambda_tone", 0.0))

    if not use_notone and not use_tone:
        raise ValueError("At least one auxiliary task must be enabled.")
    if use_notone and lambda_notone <= 0:
        raise ValueError("use_notone=true requires lambda_notone > 0.")
    if (not use_notone) and lambda_notone != 0:
        raise ValueError("use_notone=false requires lambda_notone = 0.")
    if use_tone and lambda_tone <= 0:
        raise ValueError("use_tone=true requires lambda_tone > 0.")
    if (not use_tone) and lambda_tone != 0:
        raise ValueError("use_tone=false requires lambda_tone = 0.")

    tone_regex = str(conf.get("tone_regex", r"^(.*?)([0-9]+)$"))
    unmarked_tone_token = str(conf.get("unmarked_tone_token", "<no_tone>"))
    if unmarked_tone_token in AUX_SPECIALS:
        raise ValueError(
            "unmarked_tone_token must be a dedicated non-special tone label, "
            f"but got {unmarked_tone_token!r}."
        )
    try:
        pat = re.compile(tone_regex)
    except re.error as e:
        raise ValueError(f"Invalid tone_regex={tone_regex!r}: {e}") from e
    if pat.groups < 2:
        raise ValueError(
            "tone_regex must have at least two capture groups: "
            "group(1)=no-tone syllable, group(2)=tone."
        )


def validate_training_strategy(conf):
    """Validate optional auxiliary-training strategy.

    simultaneous: original behavior, unchanged.
    delayed: epoch-wise schedule from aux_schedule.
    alternate: main is always active; no-tone/tone alternate by optimizer step.
    """
    strategy = str(conf.get("aux_training_strategy", "simultaneous")).lower()
    if strategy not in {"simultaneous", "delayed", "alternate"}:
        raise ValueError(
            "aux_training_strategy must be simultaneous, delayed, or alternate; "
            f"got {strategy!r}"
        )

    if strategy == "delayed":
        schedule = conf.get("aux_schedule")
        if not isinstance(schedule, list) or not schedule:
            raise ValueError("delayed strategy requires non-empty aux_schedule")

        starts = []
        for item in schedule:
            if not isinstance(item, dict):
                raise ValueError("each aux_schedule item must be a mapping")
            start = int(item["start_epoch"])
            wn = float(item.get("lambda_notone", 0.0))
            wt = float(item.get("lambda_tone", 0.0))
            if start < 0 or wn < 0 or wt < 0:
                raise ValueError(f"invalid aux_schedule item: {item}")
            starts.append(start)

        if starts != sorted(starts) or len(starts) != len(set(starts)):
            raise ValueError(
                "aux_schedule start_epoch values must be unique and sorted"
            )
        if starts[0] != 0:
            raise ValueError("aux_schedule must start at epoch 0")

    if strategy == "alternate":
        first = str(conf.get("alternate_first", "notone")).lower()
        if first not in {"notone", "tone"}:
            raise ValueError("alternate_first must be notone or tone")
        if not (bool(conf.get("use_notone")) and bool(conf.get("use_tone"))):
            raise ValueError(
                "alternate strategy requires both auxiliary heads enabled"
            )


def delayed_epoch_weights(conf, epoch):
    schedule = conf["aux_schedule"]
    chosen = schedule[0]
    for item in schedule:
        if int(item["start_epoch"]) <= int(epoch):
            chosen = item
        else:
            break
    return (
        float(chosen.get("lambda_notone", 0.0)),
        float(chosen.get("lambda_tone", 0.0)),
    )


def training_weights(conf, epoch, completed_optimizer_steps):
    """Return effective auxiliary weights for the current accumulation cycle.

    completed_optimizer_steps counts optimizer updates that have actually
    happened. Because it is incremented only after optimizer.step(), every
    mini-batch inside the same gradient-accumulation cycle receives the same
    auxiliary branch in alternate mode.
    """
    strategy = str(conf.get("aux_training_strategy", "simultaneous")).lower()
    base_notone = float(conf.get("lambda_notone", 0.0))
    base_tone = float(conf.get("lambda_tone", 0.0))

    if strategy == "simultaneous":
        return base_notone, base_tone, "simultaneous"

    if strategy == "delayed":
        wn, wt = delayed_epoch_weights(conf, epoch)
        return wn, wt, f"delayed({wn:g},{wt:g})"

    first = str(conf.get("alternate_first", "notone")).lower()
    even_task = first
    odd_task = "tone" if first == "notone" else "notone"
    task = (
        even_task
        if int(completed_optimizer_steps) % 2 == 0
        else odd_task
    )

    if task == "notone":
        return base_notone, 0.0, "alternate:notone"
    return 0.0, base_tone, "alternate:tone"


def eval_objective_weights(conf, epoch):
    """Weights used only to print a comparable Dev total loss.

    Main-task Dev CER is always computed only from the 871-way main branch
    and remains the checkpoint-selection criterion.
    """
    strategy = str(conf.get("aux_training_strategy", "simultaneous")).lower()

    if strategy == "delayed":
        return delayed_epoch_weights(conf, epoch)

    if strategy == "alternate":
        # Each auxiliary branch participates in about half of optimizer
        # updates. Half-weights represent that expected objective for logging.
        return (
            0.5 * float(conf.get("lambda_notone", 0.0)),
            0.5 * float(conf.get("lambda_tone", 0.0)),
        )

    return (
        float(conf.get("lambda_notone", 0.0)),
        float(conf.get("lambda_tone", 0.0)),
    )


def split_full_token(
    token: str,
    tone_pattern: re.Pattern,
    unmarked_tone_token: str = "<no_tone>",
) -> Tuple[str, str]:
    """
    Split a full syllable token into (no-tone syllable, tone).

    Examples:
        long3 -> ("long", "3")
        sang1 -> ("sang", "1")
        a     -> ("a", "<no_tone>")

    Some valid Tujia main-vocabulary symbols do not carry a numeric tone suffix.
    They must NOT be rejected: for the no-tone task the base remains the token
    itself, and for the tone task they are assigned an explicit non-blank
    <no_tone> class.
    """
    m = tone_pattern.fullmatch(token)
    if m is None:
        return token, unmarked_tone_token

    base = m.group(1)
    tone = m.group(2)
    if not base or not tone:
        raise ValueError(f"Invalid no-tone/tone split for token {token!r}")
    return base, tone


def build_auxiliary_tables(
    id2token: Dict[int, str],
    tone_regex: str,
    unmarked_tone_token: str = "<no_tone>",
):
    """
    Build deterministic auxiliary vocabularies from the full 871-symbol table.

    IDs 0/1/2 stay aligned with the main task's special-token convention.
    Remaining IDs are assigned in first-occurrence order of the full dictionary.
    """
    tone_pattern = re.compile(tone_regex)

    notone_token2id = dict(AUX_SPECIALS)
    tone_token2id = dict(AUX_SPECIALS)
    full_to_notone: Dict[str, int] = {}
    full_to_tone: Dict[str, int] = {}
    split_map: Dict[str, Tuple[str, str]] = {}

    for idx in sorted(id2token):
        token = id2token[idx]
        if token in SPECIAL_TOKENS:
            continue

        base, tone = split_full_token(token, tone_pattern, unmarked_tone_token)
        split_map[token] = (base, tone)

        if base not in notone_token2id:
            notone_token2id[base] = len(notone_token2id)
        if tone not in tone_token2id:
            tone_token2id[tone] = len(tone_token2id)

        full_to_notone[token] = notone_token2id[base]
        full_to_tone[token] = tone_token2id[tone]

    if not full_to_notone:
        raise ValueError("No non-special full syllable tokens were found.")

    return (
        notone_token2id,
        tone_token2id,
        full_to_notone,
        full_to_tone,
        split_map,
    )


def write_symbol_table(path: Path, token2id: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    id2token = {idx: token for token, idx in token2id.items()}
    with path.open("w", encoding="utf-8") as f:
        for idx in sorted(id2token):
            f.write(f"{id2token[idx]} {idx}\n")


def derive_aux_ids(
    tokens: List[str],
    token2id: Dict[str, int],
    full_to_notone: Dict[str, int],
    full_to_tone: Dict[str, int],
    key: str,
):
    # Also validates the main transcript.
    _ = text_to_ids(" ".join(tokens), token2id, key=key)

    notone_ids = []
    tone_ids = []
    for token in tokens:
        if token not in full_to_notone or token not in full_to_tone:
            raise KeyError(f"{key}: auxiliary mapping missing for token {token!r}")
        notone_ids.append(full_to_notone[token])
        tone_ids.append(full_to_tone[token])
    return notone_ids, tone_ids


def validate_list_aux_labels(
    list_path: str,
    token2id: Dict[str, int],
    full_to_notone: Dict[str, int],
    full_to_tone: Dict[str, int],
):
    rows = read_jsonl(list_path)
    for row in rows:
        tokens = row["txt"].strip().split()
        if not tokens:
            raise ValueError(f"{row['key']}: empty transcript")
        derive_aux_ids(
            tokens,
            token2id,
            full_to_notone,
            full_to_tone,
            row["key"],
        )
    return rows


class JsonlMultiCTCDataset(Dataset):
    def __init__(
        self,
        list_path: str,
        token2id: Dict[str, int],
        full_to_notone: Dict[str, int],
        full_to_tone: Dict[str, int],
        sampling_rate: int,
        root: str = ".",
    ):
        self.rows = validate_list_aux_labels(
            list_path,
            token2id,
            full_to_notone,
            full_to_tone,
        )
        self.token2id = token2id
        self.full_to_notone = full_to_notone
        self.full_to_tone = full_to_tone
        self.sampling_rate = sampling_rate
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav_path = resolve_wav_path(row["wav"], self.root)
        waveform = load_audio(wav_path, self.sampling_rate)

        ref_tokens = row["txt"].strip().split()
        main_labels = text_to_ids(row["txt"], self.token2id, key=row["key"])
        notone_labels, tone_labels = derive_aux_ids(
            ref_tokens,
            self.token2id,
            self.full_to_notone,
            self.full_to_tone,
            row["key"],
        )

        return {
            "key": row["key"],
            "waveform": waveform,
            "labels": main_labels,
            "notone_labels": notone_labels,
            "tone_labels": tone_labels,
            "ref_tokens": ref_tokens,
        }


def pad_label_sequences(items: List[dict], key: str) -> torch.Tensor:
    max_len = max(len(x[key]) for x in items)
    labels = torch.full((len(items), max_len), -100, dtype=torch.long)
    for i, x in enumerate(items):
        labels[i, : len(x[key])] = torch.tensor(x[key], dtype=torch.long)
    return labels


class MultiCTCDataCollator:
    def __init__(self, feature_extractor, sampling_rate: int):
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __call__(self, items: List[dict]):
        waves = [x["waveform"].numpy() for x in items]
        batch = self.feature_extractor(
            waves,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch["labels"] = pad_label_sequences(items, "labels")
        batch["notone_labels"] = pad_label_sequences(items, "notone_labels")
        batch["tone_labels"] = pad_label_sequences(items, "tone_labels")
        batch["keys"] = [x["key"] for x in items]
        batch["ref_tokens"] = [x["ref_tokens"] for x in items]
        return batch


class XLSRMultiTaskCTC(nn.Module):
    """
    Shared XLS-R encoder + three possible sequence-level CTC heads.

    The main head is the standard Wav2Vec2ForCTC lm_head.
    Auxiliary heads are attached to the same dropped encoder representation.
    """

    def __init__(
        self,
        model_name: str,
        main_vocab_size: int,
        notone_vocab_size: int,
        tone_vocab_size: int,
        blank_id: int,
        use_notone: bool,
        use_tone: bool,
        lambda_notone: float,
        lambda_tone: float,
    ):
        super().__init__()
        self.main_model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            vocab_size=main_vocab_size,
            pad_token_id=blank_id,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            ignore_mismatched_sizes=True,
        )

        hidden_size = int(self.main_model.config.hidden_size)
        self.notone_head = (
            nn.Linear(hidden_size, notone_vocab_size) if use_notone else None
        )
        self.tone_head = nn.Linear(hidden_size, tone_vocab_size) if use_tone else None

        self.blank_id = int(blank_id)
        self.use_notone = bool(use_notone)
        self.use_tone = bool(use_tone)
        self.lambda_notone = float(lambda_notone)
        self.lambda_tone = float(lambda_tone)

    def gradient_checkpointing_enable(self):
        self.main_model.gradient_checkpointing_enable()

    def freeze_feature_encoder(self):
        if hasattr(self.main_model, "freeze_feature_encoder"):
            self.main_model.freeze_feature_encoder()
        elif hasattr(self.main_model, "freeze_feature_extractor"):
            self.main_model.freeze_feature_extractor()
        else:
            for p in self.main_model.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False

    def _get_feat_extract_output_lengths(self, input_lengths):
        return self.main_model._get_feat_extract_output_lengths(input_lengths)

    def _ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_lengths = attention_mask.sum(-1)
        input_lengths = self._get_feat_extract_output_lengths(input_lengths).to(
            dtype=torch.long
        )

        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1).to(dtype=torch.long)

        # Early actionable error instead of a cryptic CTC failure.
        if bool((target_lengths > input_lengths).any().item()):
            raise RuntimeError(
                "CTC target is longer than XLS-R output for at least one utterance. "
                f"input_lengths={input_lengths.detach().cpu().tolist()} "
                f"target_lengths={target_lengths.detach().cpu().tolist()}"
            )

        flattened_targets = labels.masked_select(labels_mask)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

        # Hugging Face Wav2Vec2ForCTC also disables cuDNN for this CTC call.
        with torch.backends.cudnn.flags(enabled=False):
            return F.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=self.blank_id,
                reduction="mean",
                zero_infinity=True,
            )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        notone_labels: Optional[torch.Tensor] = None,
        tone_labels: Optional[torch.Tensor] = None,
        lambda_notone_override: Optional[float] = None,
        lambda_tone_override: Optional[float] = None,
    ):
        enc = self.main_model.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        hidden = self.main_model.dropout(enc.last_hidden_state)

        main_logits = self.main_model.lm_head(hidden)
        loss_main = self._ctc_loss(main_logits, labels, attention_mask)
        effective_lambda_notone = (
            self.lambda_notone
            if lambda_notone_override is None
            else float(lambda_notone_override)
        )
        effective_lambda_tone = (
            self.lambda_tone
            if lambda_tone_override is None
            else float(lambda_tone_override)
        )
        loss_total = loss_main

        loss_notone = None
        notone_logits = None
        if self.use_notone:
            if notone_labels is None:
                raise ValueError("notone_labels are required for FN-X/FJ-X")
            notone_logits = self.notone_head(hidden)
            loss_notone = self._ctc_loss(
                notone_logits,
                notone_labels,
                attention_mask,
            )
            loss_total = loss_total + effective_lambda_notone * loss_notone

        loss_tone = None
        tone_logits = None
        if self.use_tone:
            if tone_labels is None:
                raise ValueError("tone_labels are required for FA-X/FJ-X")
            tone_logits = self.tone_head(hidden)
            loss_tone = self._ctc_loss(
                tone_logits,
                tone_labels,
                attention_mask,
            )
            loss_total = loss_total + effective_lambda_tone * loss_tone

        return {
            "loss": loss_total,
            "loss_main": loss_main,
            "loss_notone": loss_notone,
            "loss_tone": loss_tone,
            "logits": main_logits,
            "notone_logits": notone_logits,
            "tone_logits": tone_logits,
            "effective_lambda_notone": effective_lambda_notone,
            "effective_lambda_tone": effective_lambda_tone,
        }

    def save_auxiliary(
        self,
        best_dir: Path,
        notone_token2id: Dict[str, int],
        tone_token2id: Dict[str, int],
        conf: dict,
    ):
        best_dir.mkdir(parents=True, exist_ok=True)

        def cpu_state(head):
            if head is None:
                return None
            return {k: v.detach().cpu() for k, v in head.state_dict().items()}

        payload = {
            "use_notone": self.use_notone,
            "use_tone": self.use_tone,
            "lambda_notone": self.lambda_notone,
            "lambda_tone": self.lambda_tone,
            "notone_head": cpu_state(self.notone_head),
            "tone_head": cpu_state(self.tone_head),
            "notone_token2id": notone_token2id,
            "tone_token2id": tone_token2id,
            "tone_regex": conf["tone_regex"],
            "unmarked_tone_token": conf.get("unmarked_tone_token", "<no_tone>"),
        }
        torch.save(payload, best_dir / "auxiliary_heads.pt")

        aux_meta = {
            "use_notone": self.use_notone,
            "use_tone": self.use_tone,
            "lambda_notone": self.lambda_notone,
            "lambda_tone": self.lambda_tone,
            "notone_vocab_size": len(notone_token2id),
            "tone_vocab_size": len(tone_token2id),
            "tone_regex": conf["tone_regex"],
            "unmarked_tone_token": conf.get("unmarked_tone_token", "<no_tone>"),
            "decode_policy": "main_871_ctc_only",
        }
        (best_dir / "auxiliary_config.json").write_text(
            json.dumps(aux_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    id2token,
    blank_id,
    use_fp16,
    lambda_notone_override=None,
    lambda_tone_override=None,
):
    model.eval()

    sums = {
        "loss": 0.0,
        "loss_main": 0.0,
        "loss_notone": 0.0,
        "loss_tone": 0.0,
    }
    batches = 0

    all_keys = []
    all_refs = []
    all_hyps = []

    autocast_enabled = bool(use_fp16 and device.type == "cuda")

    for eval_idx, batch in enumerate(loader, 1):
        keys = batch.pop("keys")
        refs = batch.pop("ref_tokens")

        input_values = batch["input_values"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        notone_labels = batch["notone_labels"].to(device, non_blocking=True)
        tone_labels = batch["tone_labels"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            out = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
                notone_labels=notone_labels,
                tone_labels=tone_labels,
                lambda_notone_override=lambda_notone_override,
                lambda_tone_override=lambda_tone_override,
            )

        sums["loss"] += float(out["loss"].detach().cpu())
        sums["loss_main"] += float(out["loss_main"].detach().cpu())
        if out["loss_notone"] is not None:
            sums["loss_notone"] += float(out["loss_notone"].detach().cpu())
        if out["loss_tone"] is not None:
            sums["loss_tone"] += float(out["loss_tone"].detach().cpu())
        batches += 1

        pred = out["logits"].argmax(dim=-1)

        input_lengths = attention_mask.sum(-1)
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)

        for i, key in enumerate(keys):
            n = int(output_lengths[i].item())
            ids = pred[i, :n].detach().cpu().tolist()
            hyp = collapse_ctc(ids, blank_id, id2token)

            all_keys.append(key)
            all_refs.append(refs[i])
            all_hyps.append(hyp)

        if eval_idx % 20 == 0 or eval_idx == len(loader):
            print(f"DEV batch={eval_idx}/{len(loader)}", flush=True)

    metrics = compute_ser_cer(all_refs, all_hyps)
    metrics["loss"] = sums["loss"] / max(batches, 1)
    metrics["loss_main"] = sums["loss_main"] / max(batches, 1)
    metrics["loss_notone"] = sums["loss_notone"] / max(batches, 1)
    metrics["loss_tone"] = sums["loss_tone"] / max(batches, 1)
    return metrics, all_keys, all_refs, all_hyps


def main():
    args = parse_args()
    conf = load_config(args.config)
    apply_cli_overrides(conf, args)
    validate_aux_config(conf)
    validate_training_strategy(conf)

    set_seed(int(conf["seed"]))

    token2id, id2token = load_symbol_table(conf["dict_path"])
    vocab_size = max(id2token) + 1
    if vocab_size != int(conf["expected_vocab_size"]):
        raise SystemExit(
            f"❌ vocab size expected={conf['expected_vocab_size']}, actual={vocab_size}"
        )

    blank_id = int(conf["blank_id"])
    if blank_id != 0 or id2token.get(0) != "<blank>":
        raise SystemExit("❌ expected CTC blank: id 0 = <blank>")

    tone_regex = str(conf.get("tone_regex", r"^(.*?)([0-9]+)$"))
    unmarked_tone_token = str(conf.get("unmarked_tone_token", "<no_tone>"))
    (
        notone_token2id,
        tone_token2id,
        full_to_notone,
        full_to_tone,
        split_map,
    ) = build_auxiliary_tables(
        id2token,
        tone_regex,
        unmarked_tone_token=unmarked_tone_token,
    )

    exp_dir = Path(conf["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_symbol_table(exp_dir / "notone_symbols.txt", notone_token2id)
    write_symbol_table(exp_dir / "tone_symbols.txt", tone_token2id)

    # Validate all transcript mappings before expensive model loading.
    train_rows = validate_list_aux_labels(
        conf["train_list"], token2id, full_to_notone, full_to_tone
    )
    dev_rows = validate_list_aux_labels(
        conf["dev_list"], token2id, full_to_notone, full_to_tone
    )
    if conf.get("test_list"):
        _ = validate_list_aux_labels(
            conf["test_list"], token2id, full_to_notone, full_to_tone
        )

    experiment_name = str(conf.get("experiment_name", "XLS-R-AUX"))

    print("=" * 76)
    print(f"{experiment_name}: XLS-R-300M Tujia SP3 auxiliary CTC")
    print("model:", conf["model_name"])
    print("train:", conf["train_list"])
    print("dev:", conf["dev_list"])
    print("main vocab:", vocab_size)
    print("no-tone vocab:", len(notone_token2id))
    print("tone vocab:", len(tone_token2id))
    print("tone classes:", [
        tok for tok, idx in sorted(tone_token2id.items(), key=lambda x: x[1])
        if tok not in AUX_SPECIALS
    ])
    unmarked_full_tokens = [
        tok for tok, (_, tone) in split_map.items()
        if tone == unmarked_tone_token
    ]
    print(
        f"unmarked/no-numeric-tone full tokens: {len(unmarked_full_tokens)} "
        f"(mapped to tone class {unmarked_tone_token})"
    )
    if unmarked_full_tokens:
        print("unmarked examples:", unmarked_full_tokens[:30])
    print("use no-tone:", bool(conf["use_notone"]),
          "weight:", float(conf["lambda_notone"]))
    print("use tone:", bool(conf["use_tone"]),
          "weight:", float(conf["lambda_tone"]))
    print(
        "aux training strategy:",
        conf.get("aux_training_strategy", "simultaneous"),
    )
    if str(conf.get("aux_training_strategy", "simultaneous")).lower() == "delayed":
        print("aux schedule:", conf["aux_schedule"])
    if str(conf.get("aux_training_strategy", "simultaneous")).lower() == "alternate":
        print("alternate first:", conf.get("alternate_first", "notone"))
    print("best checkpoint criterion: MAIN-task Dev CER")
    print("decode policy: MAIN 871-way CTC only")
    print("=" * 76)

    # Show a few mappings so a wrong regex is caught visibly.
    shown = 0
    print("auxiliary label examples:")
    for idx in sorted(id2token):
        token = id2token[idx]
        if token in split_map:
            base, tone = split_map[token]
            print(f"  {token} -> no-tone={base}, tone={tone}")
            shown += 1
            if shown >= 12:
                break

    if args.check_only:
        print("train utterances:", len(train_rows))
        print("dev utterances:", len(dev_rows))
        print("✅ auxiliary transcript mapping check passed")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("❌ XLS-R-300M training is configured for CUDA")
    print("device:", torch.cuda.get_device_name(0))

    best_dir = exp_dir / "best_model"
    best_dev_dir = exp_dir / "best_dev" / "ctc_greedy_search"
    history_path = exp_dir / "history.jsonl"

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(conf["model_name"])
    if int(feature_extractor.sampling_rate) != int(conf["sampling_rate"]):
        raise SystemExit(
            "❌ sampling rate mismatch: "
            f"feature_extractor={feature_extractor.sampling_rate}"
        )

    model = XLSRMultiTaskCTC(
        model_name=conf["model_name"],
        main_vocab_size=vocab_size,
        notone_vocab_size=len(notone_token2id),
        tone_vocab_size=len(tone_token2id),
        blank_id=blank_id,
        use_notone=bool(conf["use_notone"]),
        use_tone=bool(conf["use_tone"]),
        lambda_notone=float(conf["lambda_notone"]),
        lambda_tone=float(conf["lambda_tone"]),
    )

    if bool(conf["gradient_checkpointing"]):
        model.gradient_checkpointing_enable()

    if bool(conf["freeze_feature_encoder"]):
        model.freeze_feature_encoder()

    model.to(device)

    train_set = JsonlMultiCTCDataset(
        conf["train_list"],
        token2id,
        full_to_notone,
        full_to_tone,
        int(conf["sampling_rate"]),
        args.root,
    )
    dev_set = JsonlMultiCTCDataset(
        conf["dev_list"],
        token2id,
        full_to_notone,
        full_to_tone,
        int(conf["sampling_rate"]),
        args.root,
    )

    collator = MultiCTCDataCollator(
        feature_extractor,
        int(conf["sampling_rate"]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(conf["train_batch_size"]),
        shuffle=True,
        num_workers=int(conf["num_workers"]),
        pin_memory=True,
        collate_fn=collator,
        persistent_workers=int(conf["num_workers"]) > 0,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=int(conf["eval_batch_size"]),
        shuffle=False,
        num_workers=int(conf["num_workers"]),
        pin_memory=True,
        collate_fn=collator,
        persistent_workers=int(conf["num_workers"]) > 0,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(conf["learning_rate"]),
        weight_decay=float(conf["weight_decay"]),
    )

    accum = int(conf["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * int(conf["epochs"])
    if "warmup_steps" in conf:
        warmup_steps = int(conf["warmup_steps"])
    else:
        warmup_steps = int(
            total_steps * float(conf.get("warmup_ratio", 0.1))
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(conf["fp16"]))
    autocast_enabled = bool(conf["fp16"])

    best_cer = float("inf")
    best_epoch = -1
    no_improve = 0

    print("train utterances:", len(train_set))
    print("dev utterances:", len(dev_set))
    print("train batch:", conf["train_batch_size"])
    print("grad accum:", conf["gradient_accumulation_steps"])
    print(
        "effective batch:",
        int(conf["train_batch_size"]) * int(conf["gradient_accumulation_steps"]),
    )
    print("optimizer steps/epoch:", steps_per_epoch)
    print("total optimizer steps:", total_steps)
    print("warmup steps:", warmup_steps)
    print("lr:", conf["learning_rate"])
    print("epochs:", conf["epochs"])
    print("fp16:", conf["fp16"])
    print("gradient checkpointing:", conf["gradient_checkpointing"])
    print("freeze feature encoder:", conf["freeze_feature_encoder"])

    # Counts only optimizer updates that really occurred. This is deliberately
    # separate from batch_idx so alternate mode stays correct with grad_accum=2.
    global_optimizer_steps = 0

    for epoch in range(int(conf["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        sums = {
            "loss": 0.0,
            "loss_main": 0.0,
            "loss_notone": 0.0,
            "loss_tone": 0.0,
        }
        batch_count = 0
        optimizer_steps = 0
        epoch_weight_notone_sum = 0.0
        epoch_weight_tone_sum = 0.0
        epoch_start = time.time()

        eval_wn, eval_wt = eval_objective_weights(conf, epoch)
        strategy = str(
            conf.get("aux_training_strategy", "simultaneous")
        ).lower()

        if strategy == "alternate":
            print(
                f"AUX STRATEGY epoch={epoch:02d}: "
                "alternate per optimizer step; "
                f"base_notone={float(conf['lambda_notone']):g} "
                f"base_tone={float(conf['lambda_tone']):g} "
                f"first={conf.get('alternate_first', 'notone')}"
            )
        else:
            print(
                f"AUX STRATEGY epoch={epoch:02d}: {strategy}; "
                f"effective_notone={eval_wn:g} "
                f"effective_tone={eval_wt:g}"
            )

        for batch_idx, batch in enumerate(train_loader):
            batch.pop("keys")
            batch.pop("ref_tokens")

            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            notone_labels = batch["notone_labels"].to(device, non_blocking=True)
            tone_labels = batch["tone_labels"].to(device, non_blocking=True)

            current_wn, current_wt, current_aux_mode = training_weights(
                conf,
                epoch,
                global_optimizer_steps,
            )

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                out = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    labels=labels,
                    notone_labels=notone_labels,
                    tone_labels=tone_labels,
                    lambda_notone_override=current_wn,
                    lambda_tone_override=current_wt,
                )
                loss = out["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch} batch={batch_idx}: {loss}"
                )

            sums["loss"] += float(loss.detach().cpu())
            sums["loss_main"] += float(out["loss_main"].detach().cpu())
            if out["loss_notone"] is not None:
                sums["loss_notone"] += float(out["loss_notone"].detach().cpu())
            if out["loss_tone"] is not None:
                sums["loss_tone"] += float(out["loss_tone"].detach().cpu())
            epoch_weight_notone_sum += current_wn
            epoch_weight_tone_sum += current_wt
            batch_count += 1

            scaler.scale(loss / accum).backward()

            should_step = (
                (batch_idx + 1) % accum == 0
                or (batch_idx + 1) == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(trainable_params, float(conf["max_grad_norm"]))
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()
                optimizer.zero_grad(set_to_none=True)

                # Only advance the scheduler when AMP really performed the update.
                if new_scale >= old_scale:
                    scheduler.step()
                    optimizer_steps += 1
                    global_optimizer_steps += 1

            if (
                (batch_idx + 1) % 100 == 0
                or (batch_idx + 1) == len(train_loader)
            ):
                mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                denom = max(batch_count, 1)
                print(
                    f"epoch={epoch:02d} batch={batch_idx+1}/{len(train_loader)} "
                    f"loss={sums['loss']/denom:.4f} "
                    f"main={sums['loss_main']/denom:.4f} "
                    f"notone={sums['loss_notone']/denom:.4f} "
                    f"tone={sums['loss_tone']/denom:.4f} "
                    f"wN={current_wn:g} wT={current_wt:g} "
                    f"aux_mode={current_aux_mode} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"max_gpu_mem={mem:.2f}GB",
                    flush=True,
                )

        denom = max(batch_count, 1)
        train_metrics = {k: v / denom for k, v in sums.items()}
        train_metrics["avg_lambda_notone"] = (
            epoch_weight_notone_sum / denom
        )
        train_metrics["avg_lambda_tone"] = (
            epoch_weight_tone_sum / denom
        )

        dev_metrics, keys, refs, hyps = evaluate(
            model=model,
            loader=dev_loader,
            device=device,
            id2token=id2token,
            blank_id=blank_id,
            use_fp16=bool(conf["fp16"]),
            lambda_notone_override=eval_wn,
            lambda_tone_override=eval_wt,
        )

        elapsed = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_loss_main": train_metrics["loss_main"],
            "train_loss_notone": train_metrics["loss_notone"],
            "train_loss_tone": train_metrics["loss_tone"],
            "dev_loss": dev_metrics["loss"],
            "dev_loss_main": dev_metrics["loss_main"],
            "dev_loss_notone": dev_metrics["loss_notone"],
            "dev_loss_tone": dev_metrics["loss_tone"],
            "dev_ser": dev_metrics["ser"],
            "dev_cer": dev_metrics["cer"],
            "aux_training_strategy": strategy,
            "train_avg_lambda_notone": train_metrics["avg_lambda_notone"],
            "train_avg_lambda_tone": train_metrics["avg_lambda_tone"],
            "dev_objective_lambda_notone": eval_wn,
            "dev_objective_lambda_tone": eval_wt,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_seconds": elapsed,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"\nEPOCH {epoch:02d} "
            f"train_total={train_metrics['loss']:.4f} "
            f"train_main={train_metrics['loss_main']:.4f} "
            f"train_notone={train_metrics['loss_notone']:.4f} "
            f"train_tone={train_metrics['loss_tone']:.4f} "
            f"avg_wN={train_metrics['avg_lambda_notone']:.4f} "
            f"avg_wT={train_metrics['avg_lambda_tone']:.4f}"
        )
        print(
            f"DEV losses: total={dev_metrics['loss']:.4f} "
            f"main={dev_metrics['loss_main']:.4f} "
            f"notone={dev_metrics['loss_notone']:.4f} "
            f"tone={dev_metrics['loss_tone']:.4f} "
            f"objective_wN={eval_wn:.4f} "
            f"objective_wT={eval_wt:.4f}"
        )
        print(format_ser(dev_metrics))
        print(format_cer(dev_metrics))

        if dev_metrics["cer"] < best_cer:
            best_cer = float(dev_metrics["cer"])
            best_epoch = epoch
            no_improve = 0

            best_dir.mkdir(parents=True, exist_ok=True)

            # Save the standard main XLS-R CTC model for unchanged decoding.
            model.main_model.save_pretrained(best_dir, safe_serialization=True)
            feature_extractor.save_pretrained(best_dir)

            # Keep auxiliary heads/vocabularies too, for later analysis/fusion work.
            model.save_auxiliary(
                best_dir,
                notone_token2id,
                tone_token2id,
                conf,
            )

            write_text(best_dev_dir / "text", keys, hyps)
            write_text(best_dev_dir / "ref.text", keys, refs)
            (best_dev_dir / "ser").write_text(
                format_ser(dev_metrics) + "\n", encoding="utf-8"
            )
            (best_dev_dir / "cer").write_text(
                format_cer(dev_metrics) + "\n", encoding="utf-8"
            )

            metadata = {
                "experiment_name": experiment_name,
                "best_epoch": best_epoch,
                "best_dev_cer": best_cer,
                "best_dev_ser": float(dev_metrics["ser"]),
                "model_name": conf["model_name"],
                "main_vocab_size": vocab_size,
                "notone_vocab_size": len(notone_token2id),
                "tone_vocab_size": len(tone_token2id),
                "blank_id": blank_id,
                "decode_policy": "main_871_ctc_only",
                "config": conf,
            }
            (exp_dir / "best_metrics.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"✅ new best MAIN Dev CER: {best_cer:.2f}% (epoch {best_epoch})")
        else:
            no_improve += 1
            print(
                f"no MAIN Dev CER improvement: {no_improve}/"
                f"{conf['early_stopping_patience']}"
            )

        patience = int(conf["early_stopping_patience"])
        if patience > 0 and no_improve >= patience:
            print(
                f"Early stopping: no MAIN Dev CER improvement for {patience} epochs."
            )
            break

    print("\n" + "=" * 76)
    print("Training finished:", experiment_name)
    print("best epoch:", best_epoch)
    print(f"best MAIN Dev CER: {best_cer:.2f}%")
    print("best model:", best_dir)
    print("=" * 76)


if __name__ == "__main__":
    main()
