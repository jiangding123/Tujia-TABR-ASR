# TABR: FA Main-CTC N-best and Standalone Base-CTC Rescoring

This repository contains the source implementation of the tone-aware ASR
training and TABR decoding method. No recordings, reference transcripts, trained model weights, or N-best
caches are included. Full-syllable and base-syllable token-to-ID mappings and
an anonymized coverage-constrained split manifest are included. Authorized
access to restricted audio and transcripts is still required to reproduce the
full paper results.

## Implementation summary

- **Baseline**: `xlsr_baseline/train_xlsr_tujia.py` implements the main CTC
  recognizer with full tone-marked syllable labels.
- **FA training**: `xlsr_baseline/train_xlsr_aux_tujia.py` constructs a
  sequence-level auxiliary tone CTC objective. The tone suffix is separated
  deterministically from each full syllable token; frame-level alignment
  annotations are not required.
- **Standalone Base model**: `xlsr_baseline/run_xlsr_notone_main_sp3.sh`
  describes training the independent base-syllable CTC model.
- **N-best**: `xlsr_baseline/decode_xlsr_aux_nbest.py` uses prefix beam search
  on the FA **main CTC** logits and computes exact main CTC sequence scores.
- **TABR**: `xlsr_baseline/attach_notone_main_to_fa_cache.py` scores the same
  candidates with the independently trained Base CTC model.
- **Development-set search**: `xlsr_baseline/search_xlsr_rescore_weights.py`
  selects alpha without using Test references, and
  `xlsr_baseline/apply_xlsr_rescore.py` applies the fixed weights on Test.
- **Tempo augmentation**: `augmentation/make_tempo_perturb_3x.sh` applies
  **SoX `tempo -s`**, which modifies speaking rate while approximately
  preserving pitch, at rates **0.9, 1.0 and 1.1**. The script produces
  `wav.scp` and `text`; a private-data-specific step prepares training JSONL.

## TABR main-CTC-only inference

The TABR launcher uses the `--main_ctc_only` option by default for FA N-best
candidate generation. This mode **does not load `auxiliary_heads.pt`**,
instantiate the tone auxiliary layer, calculate auxiliary logits, or score
candidates with the FA tone head. It uses the trained FA encoder and **main
CTC projection only**, leaving the prefix-beam algorithm and exact main CTC
score calculation unchanged.

The FA candidate cache contains `main_score`; `tone_score` and
`notone_score` are `null`. The independent Base scorer then populates
`notone_score`. Final TABR selection uses

```text
S_TABR(y) = S_FA-main(y) + alpha * S_Base(base(y))
beta_tone = 0
score_norm = raw
```

The `--task fax` argument is kept to identify the FA experiment. In other
experiments, omitting `--main_ctc_only` retains the auxiliary-scoring mode.

### Direct FA N-best command

Run from the authorized project's `s0/` root:

```bash
python xlsr_baseline/decode_xlsr_aux_nbest.py \
  --model_dir exp/XLSR300M_FA_tone04_SP3/best_model \
  --data_list data/private/dev.data.list \
  --dict_path data/private/symbols.txt \
  --output_jsonl exp/FA_dev_main_only.jsonl \
  --task fax --main_ctc_only \
  --beam_size 10 --token_prune 20 --fp16
```

The FA checkpoint directory must contain the main CTC model and feature
extractor; the auxiliary-head checkpoint is **not required** by this mode.

### Complete Dev search and Test apply

The orchestration script
`xlsr_baseline/run_fa_plus_notone_main_decoupled_rescore_sp3.sh`
passes `--main_ctc_only` at both Dev and Test N-best generation. It validates
cache metadata and separates main-CTC-only caches from other scoring modes.

```bash
stage=0 stop_stage=2 \
  bash xlsr_baseline/run_fa_plus_notone_main_decoupled_rescore_sp3.sh

stage=3 stop_stage=3 \
  bash xlsr_baseline/run_fa_plus_notone_main_decoupled_rescore_sp3.sh
```

The 17-value Dev alpha grid is:

```text
0,0.05,0.1,0.2,0.3,0.5,0.8,1,1.5,2,2.5,3,4,5,6,8,10
```

The Dev search program uses `--task fnx` to search the external Base weight
alpha (the N-best generator itself uses `--task fax`). Alpha is selected
on Dev only; the paper's selected setting is `alpha=2.0` with `beta_tone=0`.
For other seed pairings in the fixed-alpha experiment, provide an already
chosen `best_weights.json` containing alpha=2.0 and apply it directly; do
not rerun an independent Test-based weight search.

The project's historical metric-file name `cer` denotes **Pinyin Character
Error Rate (PCER)** in the manuscript, provided the same character
normalization is used.

## Reproducibility checks using synthetic data

This package contains tests without private speech or dictionary files:

```bash
python examples/synthetic_mapping_test.py
python examples/test_main_ctc_only_equivalence.py
bash -n xlsr_baseline/run_fa_plus_notone_main_decoupled_rescore_sp3.sh
bash -n augmentation/make_tempo_perturb_3x.sh
```

`test_main_ctc_only_equivalence.py` uses a synthetic FA model to check that
both decoder modes yield identical N-best candidate sequences, ranks, and
main CTC scores, while the main-only path does not access the auxiliary
checkpoint. It does not substitute for a real-audio model regression test.

For an additional comparison using authorized real FA audio and checkpoints,
run the original auxiliary-scored decoder and the new main-only mode on the
same input and compare their two N-best caches:

```bash
python examples/verify_fa_main_ctc_cache.py \
  --reference-cache /path/to/aux_scored_dev.jsonl \
  --main-cache /path/to/main_only_dev.jsonl
```

This verifies identical candidate ordering and main CTC scores, allowing a
strict absolute tolerance of 1e-5 for floating-point differences. A result
computed with different model weights, preprocessing, precision, beam size,
or token-pruning setting is not a valid comparison.

## Configurations, dependencies and data restrictions

The `data/private/train.data.list`, `dev.data.list`, `test.data.list` and `symbols.txt` paths in the configuration examples and launchers are **placeholders**, not files included in this repository. Replace them with local authorized manifests and a local symbol table. The Base training launcher accepts `TRAIN_LIST_FULL`, `DEV_LIST_FULL`, `TEST_LIST_FULL` and `FULL_DICT` environment overrides; the TABR launcher accepts `DEV_LIST`, `TEST_LIST` and `MAIN_DICT` overrides. These launchers read existing dataset partitions; they do not create a Train/Dev/Test split.


The included YAML examples document the Baseline and FA hyperparameters,
including `lambda_tone=0.4` for FA, and require replacing data/checkpoint
paths with authorized local inputs. The full code path uses the scripts in
`xlsr_baseline/` together with a compatible XLS-R model, PyTorch,
Transformers, Torchaudio, NumPy and SoX. Environment versions should be
pinned before archival release.

The original speech recordings and reference transcripts are not provided.
The public label dictionaries and anonymized split metadata are described in
`REPRODUCIBILITY.md`. Model checkpoints are not hosted publicly and may be
requested from the corresponding author, subject to permissions. Consequently,
this release does not independently reproduce complete experimental tables
without authorized evaluation resources.

## Reproducibility materials

- [Reproducibility and data availability](REPRODUCIBILITY.md)
- [Primary experiment settings](configs/experiment_settings.yaml)
- [Full-syllable output dictionary](labels/tujia_syllable.txt)
- [Base-syllable output dictionary](labels/dict_base.txt)
- [Anonymized coverage-constrained assignments](splits/coverage_constrained.csv)
- [Public package integrity checks](checks/verify_public_release.py)

These files do not grant access to the underlying speech corpus. Cite the
actual GitHub release tag or full commit SHA corresponding to the publication.
