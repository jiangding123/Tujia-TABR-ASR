#!/usr/bin/env python3
"""
Apply the Dev-selected auxiliary CTC weights to Test.
No Test-set weight search is performed.
"""
import argparse
import json
from pathlib import Path

from common import (
    compute_ser_cer,
    format_cer,
    format_ser,
    write_text,
    write_utterance_wer,
)


def safe_score(x):
    if x is None:
        return -float("inf")

    return float(x)


def combined_score(
    candidate,
    alpha,
    beta,
    score_norm,
):
    main = safe_score(
        candidate["main_score"]
    )
    notone = safe_score(
        candidate.get("notone_score")
    )
    tone = safe_score(
        candidate.get("tone_score")
    )

    if (
        alpha != 0.0
        and notone == -float("inf")
    ):
        return -float("inf")

    if (
        beta != 0.0
        and tone == -float("inf")
    ):
        return -float("inf")

    score = main

    if alpha != 0.0:
        score += alpha * notone

    if beta != 0.0:
        score += beta * tone

    if score_norm == "token":
        score /= max(
            len(candidate["tokens"]),
            1,
        )

    return score


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--cache", required=True
    )
    p.add_argument(
        "--weights", required=True
    )
    p.add_argument(
        "--output_dir", required=True
    )

    args = p.parse_args()

    weights = json.loads(
        Path(args.weights).read_text(
            encoding="utf-8"
        )
    )

    alpha = float(
        weights["alpha_notone"]
    )
    beta = float(
        weights["beta_tone"]
    )
    score_norm = str(
        weights.get(
            "score_norm", "raw"
        )
    )

    rows = []

    with open(
        args.cache,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    keys = []
    refs = []
    hyps = []
    selected = []

    for row in rows:
        ranked = [
            (
                combined_score(
                    candidate,
                    alpha,
                    beta,
                    score_norm,
                ),
                candidate,
            )
            for candidate
            in row["candidates"]
        ]

        ranked.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        score, best = ranked[0]

        keys.append(row["key"])
        refs.append(
            row["ref_tokens"]
        )
        hyps.append(
            best["tokens"]
        )

        selected.append({
            "key": row["key"],
            "tokens": best["tokens"],
            "final_score": score,
            "main_score": (
                best["main_score"]
            ),
            "notone_score": (
                best.get(
                    "notone_score"
                )
            ),
            "tone_score": (
                best.get(
                    "tone_score"
                )
            ),
            "original_beam_rank": (
                best.get("rank")
            ),
        })

    metrics = compute_ser_cer(
        refs, hyps
    )

    out = Path(args.output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_text(
        out / "text",
        keys,
        hyps,
    )
    write_text(
        out / "ref.text",
        keys,
        refs,
    )
    write_utterance_wer(
        out / "wer",
        keys,
        refs,
        hyps,
    )

    (out / "ser").write_text(
        format_ser(metrics) + "\n",
        encoding="utf-8",
    )
    (out / "cer").write_text(
        format_cer(metrics) + "\n",
        encoding="utf-8",
    )

    with (
        out
        / "selected_candidates.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for item in selected:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "alpha_notone": alpha,
        "beta_tone": beta,
        "score_norm": score_norm,
        "ser": metrics["ser"],
        "cer": metrics["cer"],
        "cer_acc": (
            metrics["cer_acc"]
        ),
        "weights_source": (
            args.weights
        ),
    }

    (
        out / "rescore_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(format_ser(metrics))
    print(format_cer(metrics))


if __name__ == "__main__":
    main()
