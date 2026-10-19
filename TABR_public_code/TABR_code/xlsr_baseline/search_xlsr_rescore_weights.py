#!/usr/bin/env python3
"""
Search auxiliary CTC decoding weights on DEV only.
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


def parse_grid(s):
    values = []

    for raw in s.split(","):
        raw = raw.strip()

        if raw:
            values.append(float(raw))

    if not values:
        raise ValueError("empty weight grid")

    return values


def read_cache(path):
    rows = []

    with open(
        path, "r", encoding="utf-8"
    ) as f:
        for lineno, line in enumerate(
            f, 1
        ):
            if not line.strip():
                continue

            row = json.loads(line)

            if not row.get("candidates"):
                raise ValueError(
                    f"{path}:{lineno}: "
                    "no candidates for "
                    f"{row.get('key')}"
                )

            rows.append(row)

    return rows


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


def select(
    rows,
    alpha,
    beta,
    score_norm,
):
    keys = []
    refs = []
    hyps = []
    selected = []

    for row in rows:
        ranked = []

        for candidate in row["candidates"]:
            score = combined_score(
                candidate,
                alpha,
                beta,
                score_norm,
            )

            ranked.append(
                (score, candidate)
            )

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
                best.get("notone_score")
            ),
            "tone_score": (
                best.get("tone_score")
            ),
            "original_beam_rank": (
                best.get("rank")
            ),
        })

    metrics = compute_ser_cer(
        refs, hyps
    )

    return (
        metrics,
        keys,
        refs,
        hyps,
        selected,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--cache", required=True
    )
    p.add_argument(
        "--output_dir", required=True
    )
    p.add_argument(
        "--task",
        choices=["fnx", "fax", "fjx"],
        required=True,
    )
    p.add_argument(
        "--alpha_grid",
        default=(
            "0,0.05,0.1,0.2,0.3,0.5"
        ),
    )
    p.add_argument(
        "--beta_grid",
        default=(
            "0,0.05,0.1,0.2,0.3,0.5"
        ),
    )
    p.add_argument(
        "--score_norm",
        choices=["raw", "token"],
        default="raw",
        help=(
            "raw = raw CTC sequence log-prob; "
            "token = divide final combined score "
            "by output token length"
        ),
    )

    args = p.parse_args()

    rows = read_cache(args.cache)

    alphas = parse_grid(
        args.alpha_grid
    )
    betas = parse_grid(
        args.beta_grid
    )

    if args.task == "fnx":
        betas = [0.0]

    elif args.task == "fax":
        alphas = [0.0]

    out = Path(args.output_dir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_path = (
        out / "weight_search.tsv"
    )

    results = []

    with table_path.open(
        "w", encoding="utf-8"
    ) as f:
        f.write(
            "alpha_notone\t"
            "beta_tone\t"
            "SER\tCER\tACC\n"
        )

        for alpha in alphas:
            for beta in betas:
                metrics, *_ = select(
                    rows,
                    alpha,
                    beta,
                    args.score_norm,
                )

                results.append(
                    (alpha, beta, metrics)
                )

                f.write(
                    f"{alpha:.6g}\t"
                    f"{beta:.6g}\t"
                    f"{metrics['ser']:.6f}\t"
                    f"{metrics['cer']:.6f}\t"
                    f"{metrics['cer_acc']:.6f}"
                    "\n"
                )

                print(
                    f"alpha={alpha:.3g} "
                    f"beta={beta:.3g} "
                    f"Dev SER="
                    f"{metrics['ser']:.2f}% "
                    f"CER="
                    f"{metrics['cer']:.2f}%"
                )

    # Primary criterion: minimum Dev CER.
    # Tie-break: minimum Dev SER.
    # If still tied, prefer smaller total
    # auxiliary decoding weight.
    (
        best_alpha,
        best_beta,
        _,
    ) = min(
        results,
        key=lambda x: (
            x[2]["cer"],
            x[2]["ser"],
            abs(x[0]) + abs(x[1]),
            x[0],
            x[1],
        ),
    )

    (
        metrics,
        keys,
        refs,
        hyps,
        selected,
    ) = select(
        rows,
        best_alpha,
        best_beta,
        args.score_norm,
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
        "w", encoding="utf-8"
    ) as f:
        for item in selected:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    best = {
        "task": args.task,
        "alpha_notone": (
            best_alpha
        ),
        "beta_tone": (
            best_beta
        ),
        "score_norm": (
            args.score_norm
        ),
        "selection_criterion": (
            "minimum Dev CER, then Dev SER"
        ),
        "dev_ser": metrics["ser"],
        "dev_cer": metrics["cer"],
        "dev_cer_acc": (
            metrics["cer_acc"]
        ),
        "cache": args.cache,
    }

    (
        out / "best_weights.json"
    ).write_text(
        json.dumps(
            best,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print(
        "BEST DEV RESCORING WEIGHTS"
    )
    print("task:", args.task.upper())
    print(
        "alpha_notone:",
        best_alpha,
    )
    print(
        "beta_tone:",
        best_beta,
    )
    print(
        "score_norm:",
        args.score_norm,
    )
    print(format_ser(metrics))
    print(format_cer(metrics))
    print(
        "weights:",
        out / "best_weights.json",
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
