# Public output-label dictionaries

- `tujia_syllable.txt`: 868 full-syllable classes + 3 framework-reserved IDs = 871.
- `dict_base.txt`: 325 base-syllable classes + 3 framework-reserved IDs = 328.

Both use `<token> <integer ID>` lines, with reserved IDs 0--2.
The complete token-to-ID assignments are provided, not per-audio transcripts.
The Base dictionary was checked against the first-occurrence de-toning rule
implemented in `xlsr_baseline/run_xlsr_notone_main_sp3.sh`.

Publication of these dictionaries does not authorize distribution of
utterance-level audio or reference transcripts.
