#!/usr/bin/env python3
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio


SPECIAL_TOKENS = {"<blank>", "<unk>", "<sos/eos>", "<sos>", "<eos>", "<pad>"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_symbol_table(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Supports both common WeNet formats:
      token id
      id token
    Extra columns after the first two are ignored.
    """
    token2id: Dict[str, int] = {}
    id2token: Dict[int, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: expected at least 2 columns")

            a, b = parts[0], parts[1]
            try:
                idx = int(a)
                token = b
            except ValueError:
                token = a
                try:
                    idx = int(b)
                except ValueError as e:
                    raise ValueError(
                        f"{path}:{lineno}: cannot determine token/id columns: {line}"
                    ) from e

            if token in token2id:
                raise ValueError(f"{path}:{lineno}: duplicate token {token}")
            if idx in id2token:
                raise ValueError(f"{path}:{lineno}: duplicate id {idx}")

            token2id[token] = idx
            id2token[idx] = token

    if not id2token:
        raise ValueError(f"Empty symbol table: {path}")

    max_id = max(id2token)
    missing = sorted(set(range(max_id + 1)) - set(id2token))
    if missing:
        raise ValueError(
            f"Symbol IDs must be contiguous 0..{max_id}; missing examples: {missing[:20]}"
        )

    return token2id, id2token


def read_jsonl(path: str) -> List[dict]:
    rows = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for field in ("key", "wav", "txt"):
                if field not in row:
                    raise KeyError(f"{path}:{lineno}: missing {field}")
            if row["key"] in seen:
                raise ValueError(f"{path}:{lineno}: duplicate key {row['key']}")
            seen.add(row["key"])
            rows.append(row)
    return rows


def resolve_wav_path(wav: str, root: str = ".") -> Path:
    p = Path(wav)
    if p.is_absolute():
        return p
    return Path(root) / p


def load_audio(wav_path: Path, target_sr: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.ndim != 2:
        raise RuntimeError(f"Unexpected waveform shape {waveform.shape}: {wav_path}")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).contiguous().float()


def text_to_ids(txt: str, token2id: Dict[str, int], key: str = "") -> List[int]:
    tokens = txt.strip().split()
    ids = []
    for token in tokens:
        if token not in token2id:
            raise KeyError(f"{key}: token not in symbol table: {token}")
        ids.append(token2id[token])
    if not ids:
        raise ValueError(f"{key}: empty transcript")
    return ids


def collapse_ctc(
    frame_ids: Sequence[int],
    blank_id: int,
    id2token: Dict[int, str],
) -> List[str]:
    result: List[str] = []
    prev = None
    for raw_id in frame_ids:
        idx = int(raw_id)
        if idx == prev:
            continue
        prev = idx
        if idx == blank_id:
            continue
        token = id2token.get(idx, "<unk>")
        # Keep scoring behavior close to the existing WeNet text output:
        # special symbols are not written to hypothesis text.
        if token in SPECIAL_TOKENS:
            continue
        result.append(token)
    return result


def edit_counts(ref: Sequence, hyp: Sequence) -> Tuple[int, int, int]:
    """Return substitution, deletion, insertion counts."""
    n, m = len(ref), len(hyp)
    # cell = (cost, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]

    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue

            sub = prev[j - 1]
            dele = prev[j]
            ins = cur[j - 1]

            candidates = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
            ]
            # Deterministic tie break: S, D, I.
            candidates.sort(key=lambda x: (x[0], x[2] + x[3], x[3], x[1]))
            cur.append(candidates[0])
        prev = cur

    _, s, d, ins = prev[m]
    return s, d, ins


def corpus_error_counts(
    refs: Iterable[Sequence],
    hyps: Iterable[Sequence],
) -> Dict[str, int]:
    N = C = S = D = I = 0
    for ref, hyp in zip(refs, hyps):
        s, d, ins = edit_counts(ref, hyp)
        n = len(ref)
        c = n - s - d
        N += n
        C += c
        S += s
        D += d
        I += ins
    return {"N": N, "C": C, "S": S, "D": D, "I": I}


def compute_ser_cer(
    ref_tokens: List[List[str]],
    hyp_tokens: List[List[str]],
) -> Dict[str, float]:
    ser_counts = corpus_error_counts(ref_tokens, hyp_tokens)

    ref_chars = [list("".join(x)) for x in ref_tokens]
    hyp_chars = [list("".join(x)) for x in hyp_tokens]
    cer_counts = corpus_error_counts(ref_chars, hyp_chars)

    def rate(x):
        return 100.0 * (x["S"] + x["D"] + x["I"]) / max(x["N"], 1)

    ser = rate(ser_counts)
    cer = rate(cer_counts)
    cer_acc = 100.0 * cer_counts["C"] / max(cer_counts["N"], 1)

    return {
        "ser": ser,
        "cer": cer,
        "cer_acc": cer_acc,
        "ser_counts": ser_counts,
        "cer_counts": cer_counts,
    }


def format_ser(metrics: Dict[str, float]) -> str:
    x = metrics["ser_counts"]
    return (
        f"Overall -> {metrics['ser']:.2f} % "
        f"N={x['N']} C={x['C']} S={x['S']} D={x['D']} I={x['I']}"
    )


def format_cer(metrics: Dict[str, float]) -> str:
    x = metrics["cer_counts"]
    return (
        f"CER: {metrics['cer']:.2f} % "
        f"N={x['N']} C={x['C']} S={x['S']} D={x['D']} I={x['I']} "
        f"ACC={metrics['cer_acc']:.2f} %"
    )


def write_text(path: Path, keys: Sequence[str], token_lists: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key, tokens in zip(keys, token_lists):
            if tokens:
                f.write(f"{key} {' '.join(tokens)}\n")
            else:
                f.write(f"{key}\n")


def align_tokens(
    ref: Sequence[str],
    hyp: Sequence[str],
) -> Dict[str, object]:
    """
    Levenshtein alignment with the same deterministic tie-breaking philosophy
    as edit_counts(). Returns aligned ref/hyp token streams and N/C/S/D/I.

    Alignment symbol:
      C = correct
      S = substitution
      D = deletion
      I = insertion
    """
    n, m = len(ref), len(hyp)

    # Each cell stores:
    # (cost, S, D, I, backtrace_op)
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0, None)

    for i in range(1, n + 1):
        prev = dp[i - 1][0]
        dp[i][0] = (
            prev[0] + 1,
            prev[1],
            prev[2] + 1,
            prev[3],
            "D",
        )

    for j in range(1, m + 1):
        prev = dp[0][j - 1]
        dp[0][j] = (
            prev[0] + 1,
            prev[1],
            prev[2],
            prev[3] + 1,
            "I",
        )

    def rank_tuple(cell):
        # Match the tie-breaking used by edit_counts:
        # total cost, fewer D+I, fewer I, fewer S.
        return (
            cell[0],
            cell[2] + cell[3],
            cell[3],
            cell[1],
        )

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                prev = dp[i - 1][j - 1]
                dp[i][j] = (
                    prev[0],
                    prev[1],
                    prev[2],
                    prev[3],
                    "C",
                )
                continue

            sub = dp[i - 1][j - 1]
            dele = dp[i - 1][j]
            ins = dp[i][j - 1]

            candidates = [
                (
                    sub[0] + 1,
                    sub[1] + 1,
                    sub[2],
                    sub[3],
                    "S",
                ),
                (
                    dele[0] + 1,
                    dele[1],
                    dele[2] + 1,
                    dele[3],
                    "D",
                ),
                (
                    ins[0] + 1,
                    ins[1],
                    ins[2],
                    ins[3] + 1,
                    "I",
                ),
            ]

            candidates.sort(key=rank_tuple)
            dp[i][j] = candidates[0]

    aligned_ref = []
    aligned_hyp = []
    ops = []

    i, j = n, m
    while i > 0 or j > 0:
        op = dp[i][j][4]

        if op in ("C", "S"):
            aligned_ref.append(str(ref[i - 1]))
            aligned_hyp.append(str(hyp[j - 1]))
            ops.append(op)
            i -= 1
            j -= 1

        elif op == "D":
            aligned_ref.append(str(ref[i - 1]))
            aligned_hyp.append("")
            ops.append(op)
            i -= 1

        elif op == "I":
            aligned_ref.append("")
            aligned_hyp.append(str(hyp[j - 1]))
            ops.append(op)
            j -= 1

        else:
            raise RuntimeError(
                f"invalid alignment traceback at i={i}, j={j}: {op}"
            )

    aligned_ref.reverse()
    aligned_hyp.reverse()
    ops.reverse()

    s = sum(x == "S" for x in ops)
    d = sum(x == "D" for x in ops)
    ins = sum(x == "I" for x in ops)
    c = sum(x == "C" for x in ops)

    # Sanity check against the existing corpus scorer's per-utterance logic.
    s2, d2, i2 = edit_counts(ref, hyp)
    if (s, d, ins) != (s2, d2, i2):
        raise RuntimeError(
            "detailed alignment counts differ from edit_counts: "
            f"detail={(s, d, ins)} scorer={(s2, d2, i2)}"
        )

    return {
        "N": n,
        "C": c,
        "S": s,
        "D": d,
        "I": ins,
        "aligned_ref": aligned_ref,
        "aligned_hyp": aligned_hyp,
        "ops": ops,
    }


def format_aligned_token_lines(
    aligned_ref: Sequence[str],
    aligned_hyp: Sequence[str],
    uppercase: bool = True,
) -> Tuple[str, str]:
    """
    Format lab/rec lines in a readable WeNet-like aligned style.
    """
    if len(aligned_ref) != len(aligned_hyp):
        raise ValueError("aligned_ref/aligned_hyp length mismatch")

    ref_cells = []
    hyp_cells = []

    for r, h in zip(aligned_ref, aligned_hyp):
        rr = r.upper() if uppercase else r
        hh = h.upper() if uppercase else h
        width = max(len(rr), len(hh), 1)
        ref_cells.append(rr.ljust(width))
        hyp_cells.append(hh.ljust(width))

    return " ".join(ref_cells).rstrip(), " ".join(hyp_cells).rstrip()


def write_utterance_wer(
    path: Path,
    keys: Sequence[str],
    refs: Sequence[Sequence[str]],
    hyps: Sequence[Sequence[str]],
    uppercase: bool = True,
) -> None:
    """
    Write per-utterance syllable-level error details:

      utt: ...
      WER: ... N=... C=... S=... D=... I=...
      lab: ...
      rec: ...

    Here WER is the same token/syllable error rate that the global `ser`
    file reports as "Overall".
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for key, ref, hyp in zip(keys, refs, hyps):
            a = align_tokens(ref, hyp)
            err = a["S"] + a["D"] + a["I"]
            wer = 100.0 * err / max(a["N"], 1)

            lab_line, rec_line = format_aligned_token_lines(
                a["aligned_ref"],
                a["aligned_hyp"],
                uppercase=uppercase,
            )

            f.write(f"utt: {key}\n")
            f.write(
                f"WER: {wer:.2f} % "
                f"N={a['N']} C={a['C']} S={a['S']} "
                f"D={a['D']} I={a['I']}\n"
            )
            f.write(f"lab: {lab_line}\n")
            f.write(f"rec: {rec_line}\n\n")
