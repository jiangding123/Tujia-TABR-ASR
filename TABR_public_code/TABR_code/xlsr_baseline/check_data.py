#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

import torchaudio

from common import load_symbol_table, read_jsonl, resolve_wav_path, text_to_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_list", required=True)
    p.add_argument("--dev_list", required=True)
    p.add_argument("--test_list", required=True)
    p.add_argument("--dict_path", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--expected_vocab_size", type=int, default=871)
    p.add_argument("--sampling_rate", type=int, default=16000)
    args = p.parse_args()

    token2id, id2token = load_symbol_table(args.dict_path)
    vocab_size = max(id2token) + 1

    if vocab_size != args.expected_vocab_size:
        raise SystemExit(
            f"❌ vocab size expected={args.expected_vocab_size}, actual={vocab_size}"
        )
    for idx, expected in [(0, "<blank>"), (1, "<unk>"), (2, "<sos/eos>")]:
        actual = id2token.get(idx)
        if actual != expected:
            raise SystemExit(
                f"❌ special token mismatch: id={idx}, expected={expected}, actual={actual}"
            )

    print("✅ vocabulary:", vocab_size)
    print("special:", {i: id2token[i] for i in (0, 1, 2)})

    # Derive split sizes from the user-supplied manifests. No corpus-size
    # assumptions or private split statistics are embedded in this release.
    for name, path in [
        ("train", args.train_list),
        ("dev", args.dev_list),
        ("test", args.test_list),
    ]:
        rows = read_jsonl(path)
        print(f"\n{name}: {len(rows)}")

        speed = Counter()
        total_tokens = 0
        missing_wav = []
        bad_sr = []
        max_seconds = 0.0
        max_item = None

        for row in rows:
            key = row["key"]
            text_to_ids(row["txt"], token2id, key=key)
            total_tokens += len(row["txt"].split())

            for suffix in ("sp09", "sp10", "sp11"):
                if key.endswith("_" + suffix):
                    speed[suffix] += 1

            wav = resolve_wav_path(row["wav"], args.root)
            if not wav.is_file():
                missing_wav.append(str(wav))
                continue

            info = torchaudio.info(str(wav))
            if info.sample_rate != args.sampling_rate:
                bad_sr.append((key, info.sample_rate))

            sec = info.num_frames / max(info.sample_rate, 1)
            if sec > max_seconds:
                max_seconds = sec
                max_item = key

        if missing_wav:
            raise SystemExit(
                f"❌ {name}: missing wav files, examples={missing_wav[:5]}"
            )

        print("tokens:", total_tokens)
        if speed:
            print("speed:", dict(speed))
        print(f"max duration: {max_seconds:.2f}s ({max_item})")
        if bad_sr:
            print(
                f"⚠️ {name}: {len(bad_sr)} files are not {args.sampling_rate} Hz; "
                "the XLS-R loader will resample them."
            )

    print("\n✅ XLS-R data preflight passed")


if __name__ == "__main__":
    main()
