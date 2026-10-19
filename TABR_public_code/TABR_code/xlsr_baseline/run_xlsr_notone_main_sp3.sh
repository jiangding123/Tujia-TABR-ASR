#!/bin/bash
set -euo pipefail

# Standalone no-tone MAIN-task XLS-R-300M training for Tujia.
#
# Purpose:
#   Train an independent XLS-R model whose ONLY CTC task is the tone-stripped
#   syllable sequence, e.g.:
#       zong1 long3 a  ->  zong long a
#
# This is intentionally NOT FN auxiliary training:
#   - no 871-way full-tone main loss
#   - no auxiliary head
#   - the normal Hugging Face lm_head itself is the no-tone CTC head
#
# It reuses the existing baseline trainer/decoder/scorer:
#   xlsr_baseline/train_xlsr_tujia.py
#   xlsr_baseline/decode_xlsr_tujia.py
#   xlsr_baseline/score_asr_text.py
#
# Run from:
#   s0
#
# Put this file at:
#   s0/xlsr_baseline/run_xlsr_notone_main_sp3.sh
#
# Stages:
#   0: derive no-tone vocabulary + tone-stripped train/dev/test JSONL + preflight
#   1: train standalone no-tone MAIN CTC model
#   2: decode no-tone Dev/Test
#   3: score no-tone Dev/Test
#
# Recommended:
#   stage=0 stop_stage=0 bash xlsr_baseline/run_xlsr_notone_main_sp3.sh
#   stage=1 stop_stage=1 bash xlsr_baseline/run_xlsr_notone_main_sp3.sh
#   stage=2 stop_stage=3 bash xlsr_baseline/run_xlsr_notone_main_sp3.sh
#
# Or all at once:
#   bash xlsr_baseline/run_xlsr_notone_main_sp3.sh

stage=${stage:-0}
stop_stage=${stop_stage:-3}

S0_ROOT=$(pwd)
SCRIPT_DIR="${S0_ROOT}/xlsr_baseline"

# Original full-tone data/dictionary.
TRAIN_LIST_FULL=${TRAIN_LIST_FULL:-data/private/train.data.list}
DEV_LIST_FULL=${DEV_LIST_FULL:-data/private/dev.data.list}
TEST_LIST_FULL=${TEST_LIST_FULL:-data/private/test.data.list}
FULL_DICT=${FULL_DICT:-data/private/symbols.txt}

# Deterministically generated standalone no-tone data/dictionary.
NOTONE_DATA_DIR=${NOTONE_DATA_DIR:-data/xlsr_notone_main}
TRAIN_LIST="${NOTONE_DATA_DIR}/train.filtered.notone.data.list"
DEV_LIST="${NOTONE_DATA_DIR}/dev.notone.data.list"
TEST_LIST="${NOTONE_DATA_DIR}/test.notone.data.list"
NOTONE_DICT="${NOTONE_DATA_DIR}/notone_symbols.txt"
NOTONE_MAP="${NOTONE_DATA_DIR}/full_to_notone.tsv"

MODEL_NAME=${MODEL_NAME:-facebook/wav2vec2-xls-r-300m}
EXP_DIR=${EXP_DIR:-exp/XLSR300M_NOTONE_MAIN_SP3}
CONFIG="${EXP_DIR}/xlsr300m_notone_main_sp3.yaml"

EXPECTED_NOTONE_VOCAB=${EXPECTED_NOTONE_VOCAB:-328}
SEED=${SEED:-777}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-30}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-"${HOME}/.cache/huggingface"}
mkdir -p "${HF_HOME}" "${EXP_DIR}"

echo "============================================================"
echo "XLS-R-300M standalone NO-TONE MAIN CTC"
echo "stage=${stage}, stop_stage=${stop_stage}"
echo "full train=${TRAIN_LIST_FULL}"
echo "full dev=${DEV_LIST_FULL}"
echo "full test=${TEST_LIST_FULL}"
echo "full dict=${FULL_DICT}"
echo "notone data dir=${NOTONE_DATA_DIR}"
echo "notone dict=${NOTONE_DICT}"
echo "exp=${EXP_DIR}"
echo "seed=${SEED}"
echo "train batch=${TRAIN_BATCH_SIZE}"
echo "grad accum=${GRAD_ACCUM}"
echo "effective batch=$((TRAIN_BATCH_SIZE * GRAD_ACCUM))"
echo "lr=${LR}"
echo "epochs=${EPOCHS}"
echo "============================================================"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "[Stage 0] Build standalone no-tone target data + vocabulary"

  for f in \
    "${TRAIN_LIST_FULL}" \
    "${DEV_LIST_FULL}" \
    "${TEST_LIST_FULL}" \
    "${FULL_DICT}" \
    "${SCRIPT_DIR}/common.py" \
    "${SCRIPT_DIR}/check_data.py" \
    "${SCRIPT_DIR}/train_xlsr_tujia.py" \
    "${SCRIPT_DIR}/decode_xlsr_tujia.py" \
    "${SCRIPT_DIR}/score_asr_text.py"; do
    if [ ! -f "${f}" ]; then
      echo "❌ Missing file: ${f}"
      exit 1
    fi
  done

  mkdir -p "${NOTONE_DATA_DIR}"

  python - \
    "${FULL_DICT}" \
    "${TRAIN_LIST_FULL}" \
    "${DEV_LIST_FULL}" \
    "${TEST_LIST_FULL}" \
    "${NOTONE_DATA_DIR}" \
    "${EXPECTED_NOTONE_VOCAB}" <<'PY'
import json
import re
import sys
from pathlib import Path

full_dict = Path(sys.argv[1])
train_in = Path(sys.argv[2])
dev_in = Path(sys.argv[3])
test_in = Path(sys.argv[4])
out_dir = Path(sys.argv[5])
expected_vocab = int(sys.argv[6])

tone_re = re.compile(r"^(.*?)([0-9]+)$")
special_tokens = {"<blank>", "<unk>", "<sos/eos>", "<sos>", "<eos>", "<pad>"}

def load_symbol_table(path):
    token2id = {}
    id2token = {}
    with path.open("r", encoding="utf-8") as f:
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
                idx = int(b)
            if token in token2id:
                raise ValueError(f"{path}:{lineno}: duplicate token {token!r}")
            if idx in id2token:
                raise ValueError(f"{path}:{lineno}: duplicate id {idx}")
            token2id[token] = idx
            id2token[idx] = token

    if not id2token:
        raise ValueError(f"empty symbol table: {path}")

    max_id = max(id2token)
    missing = sorted(set(range(max_id + 1)) - set(id2token))
    if missing:
        raise ValueError(
            f"{path}: IDs must be contiguous 0..{max_id}; missing={missing[:20]}"
        )
    return token2id, id2token

_, id2token = load_symbol_table(full_dict)

for idx, expected in [(0, "<blank>"), (1, "<unk>"), (2, "<sos/eos>")]:
    actual = id2token.get(idx)
    if actual != expected:
        raise ValueError(
            f"full dictionary special mismatch: id={idx}, "
            f"expected={expected!r}, actual={actual!r}"
        )

# IMPORTANT:
# Reproduce the exact vocabulary-construction order used by
# train_xlsr_aux_tujia.py::build_auxiliary_tables():
#   0 <blank>, 1 <unk>, 2 <sos/eos>,
# then first occurrence of each base syllable while scanning the 871-way
# full dictionary by ascending ID.
notone_token2id = {
    "<blank>": 0,
    "<unk>": 1,
    "<sos/eos>": 2,
}
full_to_base = {}

for idx in sorted(id2token):
    token = id2token[idx]
    if token in special_tokens:
        continue

    m = tone_re.fullmatch(token)
    if m is None:
        base = token
    else:
        base = m.group(1)
        tone = m.group(2)
        if not base or not tone:
            raise ValueError(f"invalid tone split: {token!r}")

    if base not in notone_token2id:
        notone_token2id[base] = len(notone_token2id)
    full_to_base[token] = base

actual_vocab = len(notone_token2id)
if actual_vocab != expected_vocab:
    raise SystemExit(
        f"❌ no-tone vocab mismatch: expected={expected_vocab}, actual={actual_vocab}. "
        "Stop here rather than training with a silently different label space."
    )

out_dir.mkdir(parents=True, exist_ok=True)

dict_out = out_dir / "notone_symbols.txt"
id2notone = {idx: tok for tok, idx in notone_token2id.items()}
with dict_out.open("w", encoding="utf-8") as f:
    for idx in range(actual_vocab):
        f.write(f"{id2notone[idx]} {idx}\n")

map_out = out_dir / "full_to_notone.tsv"
with map_out.open("w", encoding="utf-8") as f:
    f.write("full_token\tfull_id\tnotone_token\tnotone_id\n")
    for idx in sorted(id2token):
        token = id2token[idx]
        if token in special_tokens:
            continue
        base = full_to_base[token]
        f.write(
            f"{token}\t{idx}\t{base}\t{notone_token2id[base]}\n"
        )

def convert_jsonl(src, dst):
    n_rows = 0
    n_tokens = 0
    changed_tokens = 0

    with src.open("r", encoding="utf-8") as fi, \
         dst.open("w", encoding="utf-8") as fo:
        seen = set()
        for lineno, line in enumerate(fi, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for field in ("key", "wav", "txt"):
                if field not in row:
                    raise KeyError(f"{src}:{lineno}: missing {field}")

            key = row["key"]
            if key in seen:
                raise ValueError(f"{src}:{lineno}: duplicate key {key!r}")
            seen.add(key)

            full_tokens = row["txt"].strip().split()
            if not full_tokens:
                raise ValueError(f"{src}:{lineno}: empty transcript")

            bases = []
            for tok in full_tokens:
                if tok == "<unk>":
                    base = "<unk>"
                elif tok in full_to_base:
                    base = full_to_base[tok]
                else:
                    raise KeyError(
                        f"{src}:{lineno} key={key}: "
                        f"token not found in full dictionary/mapping: {tok!r}"
                    )
                bases.append(base)
                n_tokens += 1
                changed_tokens += int(base != tok)

            row["txt"] = " ".join(bases)
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1

    return n_rows, n_tokens, changed_tokens

stats = []
for src, name in [
    (train_in, "train.filtered.notone.data.list"),
    (dev_in, "dev.notone.data.list"),
    (test_in, "test.notone.data.list"),
]:
    dst = out_dir / name
    stats.append((name, *convert_jsonl(src, dst)))

print("✅ standalone no-tone data prepared")
print(f"full dictionary size: {len(id2token)}")
print(f"no-tone vocabulary size: {actual_vocab}")
print(f"dictionary: {dict_out}")
print(f"mapping: {map_out}")
for name, rows, toks, changed in stats:
    print(
        f"{name}: utterances={rows}, tokens={toks}, "
        f"tone-stripped tokens={changed}"
    )

# A few deterministic sanity examples from the full vocabulary.
shown = 0
print("mapping examples:")
for idx in sorted(id2token):
    tok = id2token[idx]
    if tok in full_to_base and tok != full_to_base[tok]:
        base = full_to_base[tok]
        print(
            f"  {tok} -> {base} "
            f"(no-tone id={notone_token2id[base]})"
        )
        shown += 1
        if shown >= 8:
            break
PY

  cat > "${CONFIG}" <<EOF
model_name: ${MODEL_NAME}
train_list: ${TRAIN_LIST}
dev_list: ${DEV_LIST}
test_list: ${TEST_LIST}
dict_path: ${NOTONE_DICT}
exp_dir: ${EXP_DIR}
sampling_rate: 16000
blank_id: 0
expected_vocab_size: ${EXPECTED_NOTONE_VOCAB}
seed: ${SEED}
epochs: ${EPOCHS}
train_batch_size: ${TRAIN_BATCH_SIZE}
eval_batch_size: ${EVAL_BATCH_SIZE}
gradient_accumulation_steps: ${GRAD_ACCUM}
learning_rate: ${LR}
weight_decay: 0.01
warmup_steps: 500
max_grad_norm: 1.0
num_workers: 2
fp16: true
gradient_checkpointing: true
freeze_feature_encoder: true
early_stopping_patience: 0
EOF

  echo
  echo "Generated config:"
  cat "${CONFIG}"
  echo

  python "${SCRIPT_DIR}/check_data.py" \
    --train_list "${TRAIN_LIST}" \
    --dev_list "${DEV_LIST}" \
    --test_list "${TEST_LIST}" \
    --dict_path "${NOTONE_DICT}" \
    --root "${S0_ROOT}" \
    --expected_vocab_size "${EXPECTED_NOTONE_VOCAB}" \
    --sampling_rate 16000

  echo
  echo "✅ Stage 0 passed."
  echo "This label space is standalone no-tone MAIN CTC, not an auxiliary head."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[Stage 1] Fine-tune XLS-R-300M on NO-TONE MAIN CTC only"

  if [ ! -f "${CONFIG}" ]; then
    echo "❌ Missing generated config: ${CONFIG}"
    echo "Run stage 0 first."
    exit 1
  fi

  python "${SCRIPT_DIR}/train_xlsr_tujia.py" \
    --config "${CONFIG}" \
    --root "${S0_ROOT}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --epochs "${EPOCHS}" \
    2>&1 | tee "${EXP_DIR}/train.log"
fi

BEST_MODEL="${EXP_DIR}/best_model"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "[Stage 2] Decode no-tone Dev/Test"

  if [ ! -f "${BEST_MODEL}/config.json" ]; then
    echo "❌ Best model does not exist: ${BEST_MODEL}"
    exit 1
  fi
  if [ ! -f "${NOTONE_DICT}" ]; then
    echo "❌ Missing no-tone dictionary: ${NOTONE_DICT}"
    exit 1
  fi

  DEV_OUT="${EXP_DIR}/decode_dev_notone/ctc_greedy_search"
  TEST_OUT="${EXP_DIR}/decode_test_notone/ctc_greedy_search"

  python "${SCRIPT_DIR}/decode_xlsr_tujia.py" \
    --model_dir "${BEST_MODEL}" \
    --data_list "${DEV_LIST}" \
    --dict_path "${NOTONE_DICT}" \
    --output_dir "${DEV_OUT}" \
    --root "${S0_ROOT}" \
    --batch_size "${EVAL_BATCH_SIZE}"

  python "${SCRIPT_DIR}/decode_xlsr_tujia.py" \
    --model_dir "${BEST_MODEL}" \
    --data_list "${TEST_LIST}" \
    --dict_path "${NOTONE_DICT}" \
    --output_dir "${TEST_OUT}" \
    --root "${S0_ROOT}" \
    --batch_size "${EVAL_BATCH_SIZE}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "[Stage 3] Score NO-TONE Dev/Test"

  DEV_OUT="${EXP_DIR}/decode_dev_notone/ctc_greedy_search"
  TEST_OUT="${EXP_DIR}/decode_test_notone/ctc_greedy_search"

  python "${SCRIPT_DIR}/score_asr_text.py" \
    --ref_list "${DEV_LIST}" \
    --hyp_text "${DEV_OUT}/text" \
    --output_dir "${DEV_OUT}"

  python "${SCRIPT_DIR}/score_asr_text.py" \
    --ref_list "${TEST_LIST}" \
    --hyp_text "${TEST_OUT}/text" \
    --output_dir "${TEST_OUT}"

  echo
  echo "================ STANDALONE NO-TONE MAIN RESULTS ================"
  echo "DEV (SER here = no-tone/base syllable error rate):"
  cat "${DEV_OUT}/ser"
  cat "${DEV_OUT}/cer"
  echo
  echo "TEST (SER here = no-tone/base syllable error rate):"
  cat "${TEST_OUT}/ser"
  cat "${TEST_OUT}/cer"
  echo
  echo "Best standalone no-tone model:"
  echo "  ${BEST_MODEL}"
  echo "No-tone dictionary:"
  echo "  ${NOTONE_DICT}"
  echo "================================================================="
fi
