#!/usr/bin/env python3
import argparse
from pathlib import Path

from common import compute_ser_cer, format_cer, format_ser, read_jsonl, write_utterance_wer


def read_hyp(path: str):
    hyp = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.rstrip("\n").split(maxsplit=1)
            if not parts:
                continue
            key = parts[0]
            tokens = parts[1].split() if len(parts) > 1 else []
            if key in hyp:
                raise ValueError(f"{path}:{lineno}: duplicate key {key}")
            hyp[key] = tokens
    return hyp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_list", required=True)
    p.add_argument("--hyp_text", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    refs = read_jsonl(args.ref_list)
    hyp = read_hyp(args.hyp_text)

    keys = [x["key"] for x in refs]
    missing = [k for k in keys if k not in hyp]
    extra = sorted(set(hyp) - set(keys))

    if missing:
        raise SystemExit(f"❌ missing hypotheses: {len(missing)}, examples={missing[:10]}")
    if extra:
        raise SystemExit(f"❌ extra hypotheses: {len(extra)}, examples={extra[:10]}")

    ref_tokens = [x["txt"].split() for x in refs]
    hyp_tokens = [hyp[k] for k in keys]

    metrics = compute_ser_cer(ref_tokens, hyp_tokens)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ser_line = format_ser(metrics)
    cer_line = format_cer(metrics)

    (out / "ser").write_text(ser_line + "\n", encoding="utf-8")
    (out / "cer").write_text(cer_line + "\n", encoding="utf-8")
    write_utterance_wer(
        out / "wer",
        keys,
        ref_tokens,
        hyp_tokens,
    )

    print(ser_line)
    print(cer_line)


if __name__ == "__main__":
    main()
