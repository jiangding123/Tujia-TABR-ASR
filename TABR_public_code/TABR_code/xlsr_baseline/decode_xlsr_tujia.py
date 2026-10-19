#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from common import (
    collapse_ctc,
    load_audio,
    load_symbol_table,
    read_jsonl,
    resolve_wav_path,
    write_text,
)


class DecodeDataset(Dataset):
    def __init__(self, list_path: str, sampling_rate: int, root: str = "."):
        self.rows = read_jsonl(list_path)
        self.sampling_rate = sampling_rate
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


class DecodeCollator:
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
        batch["keys"] = [x["key"] for x in items]
        batch["ref_tokens"] = [x["ref_tokens"] for x in items]
        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data_list", required=True)
    p.add_argument("--dict_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--sampling_rate", type=int, default=16000)
    p.add_argument("--blank_id", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("❌ decoding is configured for CUDA")
    device = torch.device("cuda")

    _, id2token = load_symbol_table(args.dict_path)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(device)
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)

    dataset = DecodeDataset(args.data_list, args.sampling_rate, args.root)
    collator = DecodeCollator(feature_extractor, args.sampling_rate)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        persistent_workers=args.num_workers > 0,
    )

    keys = []
    refs = []
    hyps = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch_keys = batch.pop("keys")
            batch_refs = batch.pop("ref_tokens")

            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                logits = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                ).logits

            pred = logits.argmax(dim=-1)
            input_lengths = attention_mask.sum(-1)
            output_lengths = model._get_feat_extract_output_lengths(input_lengths)

            for i, key in enumerate(batch_keys):
                n = int(output_lengths[i].item())
                frame_ids = pred[i, :n].detach().cpu().tolist()
                hyp = collapse_ctc(frame_ids, args.blank_id, id2token)

                keys.append(key)
                refs.append(batch_refs[i])
                hyps.append(hyp)

            if (idx + 1) % 50 == 0:
                print(f"decoded batches: {idx+1}/{len(loader)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # WeNet-compatible text style: "key token1 token2 ..."
    write_text(out / "text", keys, hyps)
    write_text(out / "ref.text", keys, refs)

    print("decoded utterances:", len(keys))
    print("text:", out / "text")
    print("ref:", out / "ref.text")


if __name__ == "__main__":
    main()
