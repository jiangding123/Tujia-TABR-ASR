#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    get_linear_schedule_with_warmup,
)

from common import (
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


class JsonlASRDataset(Dataset):
    def __init__(
        self,
        list_path: str,
        token2id: Dict[str, int],
        sampling_rate: int,
        root: str = ".",
    ):
        self.rows = read_jsonl(list_path)
        self.token2id = token2id
        self.sampling_rate = sampling_rate
        self.root = root

        # Validate transcripts once before training.
        for row in self.rows:
            text_to_ids(row["txt"], token2id, key=row["key"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav_path = resolve_wav_path(row["wav"], self.root)
        waveform = load_audio(wav_path, self.sampling_rate)
        labels = text_to_ids(row["txt"], self.token2id, key=row["key"])
        return {
            "key": row["key"],
            "waveform": waveform,
            "labels": labels,
            "ref_tokens": row["txt"].split(),
        }


class CTCDataCollator:
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

        max_lab = max(len(x["labels"]) for x in items)
        labels = torch.full((len(items), max_lab), -100, dtype=torch.long)
        for i, x in enumerate(items):
            labels[i, : len(x["labels"])] = torch.tensor(
                x["labels"], dtype=torch.long
            )

        batch["labels"] = labels
        batch["keys"] = [x["key"] for x in items]
        batch["ref_tokens"] = [x["ref_tokens"] for x in items]
        return batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--train_batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
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
    }
    for key, value in mapping.items():
        if value is not None:
            conf[key] = value


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    id2token,
    blank_id,
    use_fp16,
):
    model.eval()
    total_loss = 0.0
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

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            out = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )

        total_loss += float(out.loss.detach().cpu())
        batches += 1

        pred = out.logits.argmax(dim=-1)

        # Ignore logits generated from right-padding.
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
            print(
                f"DEV batch={eval_idx}/{len(loader)}",
                flush=True,
            )


    metrics = compute_ser_cer(all_refs, all_hyps)
    metrics["loss"] = total_loss / max(batches, 1)

    return metrics, all_keys, all_refs, all_hyps


def main():
    args = parse_args()
    conf = load_config(args.config)
    apply_cli_overrides(conf, args)

    set_seed(int(conf["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("❌ XLS-R-300M training is configured for CUDA")

    token2id, id2token = load_symbol_table(conf["dict_path"])
    vocab_size = max(id2token) + 1
    if vocab_size != int(conf["expected_vocab_size"]):
        raise SystemExit(
            f"❌ vocab size expected={conf['expected_vocab_size']}, actual={vocab_size}"
        )
    blank_id = int(conf["blank_id"])
    if blank_id != 0 or id2token.get(0) != "<blank>":
        raise SystemExit("❌ expected CTC blank: id 0 = <blank>")

    exp_dir = Path(conf["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    best_dir = exp_dir / "best_model"
    best_dev_dir = exp_dir / "best_dev" / "ctc_greedy_search"
    history_path = exp_dir / "history.jsonl"

    print("=" * 70)
    print("XLS-R-300M Tujia SP3 baseline")
    print("model:", conf["model_name"])
    print("device:", torch.cuda.get_device_name(0))
    print("train:", conf["train_list"])
    print("dev:", conf["dev_list"])
    print("vocab:", vocab_size)
    print("blank:", blank_id)
    print("train batch:", conf["train_batch_size"])
    print("grad accum:", conf["gradient_accumulation_steps"])
    print("effective batch:", conf["train_batch_size"] * conf["gradient_accumulation_steps"])
    print("lr:", conf["learning_rate"])
    print("epochs:", conf["epochs"])
    print("fp16:", conf["fp16"])
    print("gradient checkpointing:", conf["gradient_checkpointing"])
    print("freeze feature encoder:", conf["freeze_feature_encoder"])
    print("best checkpoint criterion: Dev CER")
    print("=" * 70)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        conf["model_name"]
    )

    if int(feature_extractor.sampling_rate) != int(conf["sampling_rate"]):
        raise SystemExit(
            f"❌ sampling rate mismatch: feature_extractor={feature_extractor.sampling_rate}"
        )

    model = Wav2Vec2ForCTC.from_pretrained(
        conf["model_name"],
        vocab_size=vocab_size,
        pad_token_id=blank_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        ignore_mismatched_sizes=True,
    )

    # The 871-way CTC head is task-specific and is randomly initialized.
    if bool(conf["gradient_checkpointing"]):
        model.gradient_checkpointing_enable()

    if bool(conf["freeze_feature_encoder"]):
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        elif hasattr(model, "freeze_feature_extractor"):
            model.freeze_feature_extractor()
        else:
            for p in model.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False

    model.to(device)

    train_set = JsonlASRDataset(
        conf["train_list"], token2id, int(conf["sampling_rate"]), args.root
    )
    dev_set = JsonlASRDataset(
        conf["dev_list"], token2id, int(conf["sampling_rate"]), args.root
    )

    collator = CTCDataCollator(feature_extractor, int(conf["sampling_rate"]))

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
            total_steps * float(
                conf.get("warmup_ratio", 0.1)
            )
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
    print("optimizer steps/epoch:", steps_per_epoch)
    print("total optimizer steps:", total_steps)
    print("warmup steps:", warmup_steps)

    for epoch in range(int(conf["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        batch_count = 0
        optimizer_steps = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch.pop("keys")
            batch.pop("ref_tokens")

            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                out = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch} batch={batch_idx}: {loss}"
                )

            running_loss += float(loss.detach().cpu())
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

                # 如果 FP16 overflow 导致 optimizer update 被跳过，
                # learning-rate scheduler 也不能前进一步。
                if new_scale >= old_scale:
                    scheduler.step()
                    optimizer_steps += 1


            if ((batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader)):

                mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(
                    f"epoch={epoch:02d} batch={batch_idx+1}/{len(train_loader)} "
                    f"loss={running_loss/batch_count:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"max_gpu_mem={mem:.2f}GB"
                )

        train_loss = running_loss / max(batch_count, 1)

        dev_metrics, keys, refs, hyps = evaluate(
            model=model,
            loader=dev_loader,
            device=device,
            id2token=id2token,
            blank_id=blank_id,
            use_fp16=bool(conf["fp16"]),
        )

        elapsed = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_ser": dev_metrics["ser"],
            "dev_cer": dev_metrics["cer"],
            "lr": scheduler.get_last_lr()[0],
            "elapsed_seconds": elapsed,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"\nEPOCH {epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f}"
        )
        print(format_ser(dev_metrics))
        print(format_cer(dev_metrics))

        if dev_metrics["cer"] < best_cer:
            best_cer = float(dev_metrics["cer"])
            best_epoch = epoch
            no_improve = 0

            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir, safe_serialization=True)
            feature_extractor.save_pretrained(best_dir)

            write_text(best_dev_dir / "text", keys, hyps)
            write_text(best_dev_dir / "ref.text", keys, refs)
            (best_dev_dir / "ser").write_text(
                format_ser(dev_metrics) + "\n", encoding="utf-8"
            )
            (best_dev_dir / "cer").write_text(
                format_cer(dev_metrics) + "\n", encoding="utf-8"
            )

            metadata = {
                "best_epoch": best_epoch,
                "best_dev_cer": best_cer,
                "best_dev_ser": float(dev_metrics["ser"]),
                "model_name": conf["model_name"],
                "vocab_size": vocab_size,
                "blank_id": blank_id,
                "config": conf,
            }
            (exp_dir / "best_metrics.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"✅ new best Dev CER: {best_cer:.2f}% (epoch {best_epoch})")
        else:
            no_improve += 1
            print(
                f"no Dev CER improvement: {no_improve}/"
                f"{conf['early_stopping_patience']}"
            )

        patience = int(conf["early_stopping_patience"])
        if patience > 0 and no_improve >= patience:
            print(
                f"Early stopping: no Dev CER improvement for {patience} epochs."
            )
            break

    print("\n" + "=" * 70)
    print("Training finished")
    print("best epoch:", best_epoch)
    print(f"best Dev CER: {best_cer:.2f}%")
    print("best model:", best_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
