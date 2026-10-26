#!/usr/bin/env python3
"""Validate a public release's label dictionaries and Table A1 split counts.

Uses only standard-library modules, public metadata and public dictionaries.
Does not access private audio/transcripts or imply that recording identity is verified.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import re

ROOT = Path(__file__).resolve().parents[1]

def symbols(path):
    parsed = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid dictionary line {path}:{n}")
        token, idx = parts[0], int(parts[1])
        if idx in parsed:
            raise ValueError(f"Repeated ID {idx} in {path}")
        parsed[idx] = token
    assert set(parsed) == set(range(len(parsed)))
    assert [parsed[i] for i in (0, 1, 2)] == ["<blank>", "<unk>", "<sos/eos>"]
    return parsed

full = symbols(ROOT / "labels/tujia_syllable.txt")
base = symbols(ROOT / "labels/dict_base.txt")
assert len(full) == 871, len(full)
assert len(base) == 328, len(base)
derived = ["<blank>", "<unk>", "<sos/eos>"]
for i in range(3, len(full)):
    token = full[i]
    match = re.fullmatch(r"(.*?)([0-9]+)", token)
    stripped = match.group(1) if match else token
    if stripped not in derived:
        derived.append(stripped)
assert [base[i] for i in range(len(base))] == derived

with (ROOT / "splits/coverage_constrained.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 2626
assert len(set(row["utterance_id"] for row in rows)) == len(rows)
assert all("TJY" not in " ".join(row.values()).upper() for row in rows)
assert Counter(x["split"] for x in rows) == {"train": 2107, "dev": 257, "test": 262}
expected_source = {
    "Natural narratives": (1398, 176, 156),
    "Book-derived speech": (709, 81, 106),
}
expected_speaker = {
    "S1": (333, 46, 58), "S2": (372, 21, 28), "S3": (376, 35, 48),
    "S4": (242, 38, 25), "S5": (208, 45, 36), "S6": (329, 27, 20),
    "S7": (247, 45, 47),
}
for column, expected in [("source_type", expected_source), ("speaker_id", expected_speaker)]:
    for key, counts in expected.items():
        observed = tuple(sum(row[column] == key and row["split"] == split for row in rows)
                         for split in ("train", "dev", "test"))
        assert observed == counts, (column, key, observed, counts)

narrative = [row for row in rows if row["source_type"] == "Natural narratives"]
book = [row for row in rows if row["source_type"] == "Book-derived speech"]
assert len(set(row["source_recording_id"] for row in narrative)) == 17
assert len(set(row["source_group_id"] for row in narrative)) == 17
assert len(set(row["source_group_id"] for row in book)) == 1
assert len(set(row["source_recording_id"] for row in book)) == 896
assert all(row["recording_id_basis"] == "book_item_proxy" for row in book)
assert all(row["recording_id_basis"] == "original_long_recording" for row in narrative)
print("PASS: split and speaker/source counts match Table A1")
print("PASS: 871-way / 328-way dictionaries and token-ID derivation match")
print("NOTE: book source-recording IDs are marked as unverified item-level proxies")
