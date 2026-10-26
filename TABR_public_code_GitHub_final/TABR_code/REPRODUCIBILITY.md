# Reproducibility and data availability

This document identifies the *actual files* in this code release. The GitHub
commit or release identifier in the paper's Data Availability Statement should
point to the exact published version of this repository.

## Experimental configurations

- `xlsr300m_sp3.yaml`: Baseline full-syllable CTC training example (seed 777).
- `xlsr300m_fa_tone04_sp3_seed777.yaml`: FA training example, including tone
  auxiliary loss and weight 0.4 (seed 777).
- `xlsr_baseline/run_xlsr_notone_main_sp3.sh`: launcher that generates the
  independent base-syllable CTC configuration.
- `xlsr_baseline/run_fa_plus_notone_main_decoupled_rescore_sp3.sh`: main-CTC
  N-best generation, independent Base scoring, and development-set alpha search.
- `configs/experiment_settings.yaml`: concise, non-executable summary of the
  reported primary settings, including reported seeds 777, 888, and 999.
  The seed-777 examples do not constitute three separately supplied seed YAMLs.

Local data paths in existing executable YAML and shell examples are
placeholders. The public full-syllable dictionary is at
`labels/tujia_syllable.txt`; for example, set `MAIN_DICT` and `FULL_DICT`
to that file when preparing an authorized environment. The Base model launcher
generates its own dictionary from the full dictionary; its token IDs should
match `labels/dict_base.txt`.

## Label mappings

The following **token-to-class-ID mappings** are included:

- `labels/tujia_syllable.txt`: 871 IDs, including three reserved symbols;
  868 full-syllable classes.
- `labels/dict_base.txt`: 328 IDs, including three reserved symbols;
  325 base-syllable classes.

The provided scripts construct base-syllable and tone-related labels from
full-syllable tokens; no manually aligned frame-level tone labels are needed.
Neither dictionary is an utterance-to-transcript file. No reference
transcriptions or audio recordings are included.

The base dictionary matches the deterministic first-occurrence ordering
produced by scanning the full dictionary in increasing ID order and removing
terminal numeric tone suffixes. To check this and the partition statistics:

```bash
python checks/verify_public_release.py
```

## Corpus composition and anonymized split metadata

`splits/coverage_constrained.csv` contains one row for each of the 2,626
utterances in the coverage-constrained partition, with anonymous utterance,
speaker, source-group and source-recording metadata plus the train/dev/test
assignment. It contains no WAV paths and no transcript text; rows are shuffled so
CSV ordering does not expose source-specific sequence numbering.

Field definitions:

- `utterance_id`: randomly assigned public identifier (not a corpus filename).
- `split`: `train`, `dev`, or `test`.
- `speaker_id`: S1--S7 as in Table A1 of the article.
- `source_type`: natural narratives or book-derived speech.
- `source_group_id`: pseudonym for the grouping identifier in the submitted
  allocation sheet; *not necessarily an individual source recording*.
- `source_recording_id`: pseudonym of an original long recording for natural
  narratives; for book-derived items it is a **one-item-per-clip proxy** under
  the paper's reported one-recording-per-utterance convention.
- `recording_id_basis`: distinguishes those two forms of recording ID.

**Book-recording provenance limit.** The submitted allocation sheet assigns
all 896 book-derived utterances one shared *batch* (`Recording`) identifier.
Table A1 reports 896 book-derived recordings. To avoid incorrectly treating
this batch code as a single source recording, the public manifest uses one
synthetic clip-level identifier per book-derived utterance, marked
`book_item_proxy`. Those identifiers are **not independently verified against
original audio filename metadata**; the allocation table alone cannot prove
896 distinct original audio recording identities. The raw batch and utterance
identifiers are not shared in this release.

Count checks against Table A1:

| Group | Train | Dev | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Natural narratives | 1,398 | 176 | 156 | 1,730 |
| Book-derived speech | 709 | 81 | 106 | 896 |
| Total | 2,107 | 257 | 262 | 2,626 |

The speaker breakdown in the manuscript is also checked automatically.
These metadata enable distribution and grouping audits, not complete
independent verification of the restricted underlying recordings.

## Evaluation and rescoring scripts

The public code contains:

- `xlsr_baseline/score_asr_text.py` and `common.py` for core recognition
  scoring (sufficient when authorized reference data and predictions exist).
- `xlsr_baseline/decode_xlsr_aux_nbest.py` for FA main-CTC N-best generation.
- `xlsr_baseline/attach_notone_main_to_fa_cache.py` for Base scoring.
- `xlsr_baseline/search_xlsr_rescore_weights.py` for Dev alpha search.
- `xlsr_baseline/apply_xlsr_rescore.py` for fixed-weight application.

The repository also includes synthetic mapping and decoder-consistency tests.
The publication's complete bootstrap, speaker-block resampling and diagnostic
result tables are **not independently reproducible from this public package
alone**, because original evaluation samples/outputs and some manuscript-
specific analysis procedures are not supplied here. Please do not interpret
these core scripts as a complete end-to-end regeneration of every table.

## Model checkpoints and restricted materials

Baseline, FA, and independent Base fine-tuned weights for seed 777 are **not
publicly hosted in this repository**. Researchers may contact the
corresponding author to request them. Any release will be evaluated under
applicable data-use agreements, institutional permissions and authorization;
availability cannot be guaranteed.

Original audio, utterance-level reference transcripts and derived N-best
caches are not distributed. Even with access to checkpoint weights, full
reproduction of the paper's numerical recognition results requires authorized
evaluation references, audio or suitably permitted derived outputs.

## Citable repository version

Use the permanent GitHub URL plus an exact **release tag or commit SHA** in
the manuscript. Commit the finished files first, then retrieve the SHA with
`git rev-parse HEAD`. The version identifier cannot be determined from a
standalone ZIP archive.
