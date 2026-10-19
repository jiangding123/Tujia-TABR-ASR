#!/usr/bin/env python3
"""Tests the ACTUAL project mapping and ranking functions on fake syllables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xlsr_baseline"))

from auxiliary_projection import split_full_token
from search_xlsr_rescore_weights import select
from apply_xlsr_rescore import combined_score

regex = r"^(.*?)([0-9]+)$"
assert split_full_token("vow1", regex, "<no_tone>") == ("vow", "1")
assert split_full_token("vow2", regex, "<no_tone>") == ("vow", "2")
assert split_full_token("bare", regex, "<no_tone>") == ("bare", "<no_tone>")
assert [split_full_token(x, regex, "<no_tone>")[0]
        for x in ("vow1", "vow2")] == ["vow", "vow"]

rows = [{
    "key": "SYNTHETIC_001",
    "ref_tokens": ["vow2"],
    "candidates": [
        {"rank": 0, "tokens": ["vow1"],
         "main_score": -0.1, "notone_score": -2.0, "tone_score": None},
        {"rank": 1, "tokens": ["vow2"],
         "main_score": -2.0, "notone_score": -0.2, "tone_score": None},
    ],
}]
at_zero, _, _, hyps_zero, _ = select(rows, 0.0, 0.0, "raw")
at_two, _, _, hyps_two, _ = select(rows, 2.0, 0.0, "raw")
assert hyps_zero == [["vow1"]]
assert hyps_two == [["vow2"]]
assert at_two["ser"] < at_zero["ser"]
for c in rows[0]["candidates"]:
    a = combined_score(c, 2.0, 0.0, "raw")
    b = c["main_score"] + 2.0 * c["notone_score"]
    assert abs(a - b) < 1e-10
print("PASS: deterministic splitting, repeated bases, and alpha=2.0 reranking.")
