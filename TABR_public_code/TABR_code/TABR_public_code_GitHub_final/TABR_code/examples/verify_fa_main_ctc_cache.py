#!/usr/bin/env python3
"""Compare legacy FA N-best and new main-CTC-only caches on identical audio.

Only the beam candidate ordering and exact FA main CTC scores must match.
Tone scores deliberately differ because main-CTC-only mode does not compute
any auxiliary scores. Intended for optional server-side real-model validation.
"""
import argparse
import json
import math
from pathlib import Path


def read_cache(path):
    result = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["key"]
        if key in result:
            raise ValueError(f"Duplicate key in {path}:{lineno}: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"Empty cache: {path}")
    return result


def verify(reference, mainonly, atol):
    old = read_cache(reference)
    new = read_cache(mainonly)
    if old.keys() != new.keys():
        raise ValueError(f"Utterance-ID mismatch: {set(old)^set(new)}")
    meta_file = mainonly.with_suffix(".meta.json")
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("decoding_mode") != "main_ctc_only":
            raise ValueError("New cache is not main_ctc_only")
    compared = 0
    for key in old:
        a, b = old[key], new[key]
        if a["ref_tokens"] != b["ref_tokens"]:
            raise AssertionError(f"Reference differs: {key}")
        cands_a, cands_b = a["candidates"], b["candidates"]
        if len(cands_a) != len(cands_b):
            raise AssertionError(f"Candidate count differs: {key}")
        for x,y in zip(cands_a,cands_b):
            if (x["rank"], x["tokens"]) != (y["rank"],y["tokens"]):
                raise AssertionError(f"Beam rank or tokens differ: {key}, rank={x['rank']}")
            xscore, yscore = x["main_score"], y["main_score"]
            if xscore is None or yscore is None:
                if xscore != yscore:
                    raise AssertionError(f"Main score missing/mismatch: {key}")
            elif not math.isclose(xscore,yscore,rel_tol=0,abs_tol=atol):
                raise AssertionError(f"Main CTC score differs: {key}, rank={x['rank']}, {xscore} vs {yscore}")
            if y.get("tone_score") is not None or y.get("notone_score") is not None:
                raise AssertionError(f"Auxiliary scores present in main-only cache: {key}")
            compared += 1
    return len(old), compared


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-cache",required=True,type=Path)
    p.add_argument("--main-cache",required=True,type=Path)
    p.add_argument("--atol",type=float,default=1e-5)
    a=p.parse_args()
    n,c=verify(a.reference_cache,a.main_cache,a.atol)
    print(f"PASS: {n} utterances, {c} candidates, matching beam ranks and FA main CTC scores")


if __name__ == "__main__":
    main()
