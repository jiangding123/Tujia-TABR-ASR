#!/bin/bash
set -euo pipefail

# Decoupled inference experiment:
#   FA model provides full-tone candidates + exact FA main CTC score.
#   Standalone NO-TONE MAIN model provides exact CTC score for base(y).
#
# Final score:
#   S(y) = S_main_FA(y) + alpha * S_notone_MAIN(base(y))
#
# alpha is searched on Dev ONLY and then frozen for Test.
# No FA tone score is used in the final score (beta=0).
#
# Prerequisite standalone model:
#   exp/XLSR300M_NOTONE_MAIN_SP3/best_model
#   data/xlsr_notone_main/notone_symbols.txt
#
# Suggested clean protocol:
#   stage=0 stop_stage=2 \
#     bash xlsr_baseline/run_fa_plus_notone_main_decoupled_rescore_sp3.sh
#
#   # inspect Dev curve / best weight, then:
#   stage=3 stop_stage=3 \
#     bash xlsr_baseline/run_fa_plus_notone_main_decoupled_rescore_sp3.sh

stage=${stage:-0}
stop_stage=${stop_stage:-3}

S0_ROOT=$(pwd)
SCRIPT_DIR="${S0_ROOT}/xlsr_baseline"

DEV_LIST=${DEV_LIST:-data/private/dev.data.list}
TEST_LIST=${TEST_LIST:-data/private/test.data.list}
MAIN_DICT=${MAIN_DICT:-data/private/symbols.txt}
NOTONE_DICT=${NOTONE_DICT:-data/xlsr_notone_main/notone_symbols.txt}

FA_MODEL_DIR=${FA_MODEL_DIR:-exp/XLSR300M_FA_tone04_SP3/best_model}
NOTONE_MODEL_DIR=${NOTONE_MODEL_DIR:-exp/XLSR300M_NOTONE_MAIN_SP3/best_model}

FA_CACHE_ROOT=${FA_CACHE_ROOT:-exp/XLSR300M_FA_tone04_SP3/aux_ctc_rescore}
OUT_ROOT=${OUT_ROOT:-exp/XLSR300M_FA_PLUS_NOTONE_MAIN_DECOUPLED}

BEAM_SIZE=${BEAM_SIZE:-10}
TOKEN_PRUNE=${TOKEN_PRUNE:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
SCORE_NORM=${SCORE_NORM:-raw}
FP16=${FP16:-1}
FORCE_CACHE=${FORCE_CACHE:-0}

# Standalone no-tone MAIN CTC can have a different raw-score scale from the
# old FN auxiliary head, so alpha is searched again from scratch.
ALPHA_GRID=${ALPHA_GRID:-"0,0.05,0.1,0.2,0.3,0.5,0.8,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0"}
ALLOW_UPPER_BOUNDARY_TEST=${ALLOW_UPPER_BOUNDARY_TEST:-0}

FA_LOCAL_CACHE_DIR="${OUT_ROOT}/fa_main_ctc_only_cache"
DEV_FA_LOCAL_CACHE="${FA_LOCAL_CACHE_DIR}/dev_nbest_scores.jsonl"
TEST_FA_LOCAL_CACHE="${FA_LOCAL_CACHE_DIR}/test_nbest_scores.jsonl"
DEV_REUSE_CACHE="${FA_CACHE_ROOT}/dev_nbest_scores.jsonl"
TEST_REUSE_CACHE="${FA_CACHE_ROOT}/test_nbest_scores.jsonl"
DEV_REUSE_META="${FA_CACHE_ROOT}/dev_nbest_scores.meta.json"
TEST_REUSE_META="${FA_CACHE_ROOT}/test_nbest_scores.meta.json"

DEV_MERGED_CACHE="${OUT_ROOT}/dev_fa_plus_notone_main_scores_main_ctc_only.jsonl"
TEST_MERGED_CACHE="${OUT_ROOT}/test_fa_plus_notone_main_scores_main_ctc_only.jsonl"
DEV_OUT="${OUT_ROOT}/dev"
TEST_OUT="${OUT_ROOT}/test"

mkdir -p "${OUT_ROOT}" "${FA_LOCAL_CACHE_DIR}"

FP16_ARG=()
if [ "${FP16}" = "1" ]; then
  FP16_ARG=(--fp16)
fi

validate_reuse_cache () {
  local meta="$1"
  local expected_model="$2"
  local expected_data="$3"

  if [ ! -s "${meta}" ]; then
    return 1
  fi

  python - "${meta}" "${expected_model}" "${expected_data}" "${BEAM_SIZE}" "${TOKEN_PRUNE}" <<'PY'
import json
import os
import sys

meta_path, expected_model, expected_data, beam_size, token_prune = sys.argv[1:]
try:
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
except Exception:
    raise SystemExit(1)

actual_model = os.path.normpath(str(meta.get("model_dir", "")))
expected_model = os.path.normpath(expected_model)
task = str(meta.get("task", "")).lower()

if actual_model != expected_model or task != "fax":
    raise SystemExit(1)
if os.path.normpath(str(meta.get("data_list", ""))) != os.path.normpath(expected_data):
    raise SystemExit(1)
if meta.get("decoding_mode") != "main_ctc_only":
    raise SystemExit(1)
if meta.get("auxiliary_checkpoint_loaded") is not False:
    raise SystemExit(1)
if meta.get("auxiliary_scoring_enabled") is not False:
    raise SystemExit(1)
if int(meta.get("beam_size", -1)) != int(beam_size):
    raise SystemExit(1)
if int(meta.get("token_prune", -1)) != int(token_prune):
    raise SystemExit(1)
PY
}

validate_merged_cache () {
  local meta="$1"
  local expected_cache="$2"
  local expected_data="$3"
  if [ ! -s "${meta}" ]; then
    return 1
  fi
  python - "${meta}" "${expected_cache}" "${expected_data}" "${NOTONE_MODEL_DIR}" <<'PY'
import json
import os
import sys

meta_path, expected_cache, expected_data, expected_model = sys.argv[1:]
try:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
except Exception:
    raise SystemExit(1)

def same_path(a, b):
    return os.path.normpath(str(a)) == os.path.normpath(str(b))

if not same_path(meta.get("base_cache"), expected_cache):
    raise SystemExit(1)
if not same_path(meta.get("data_list"), expected_data):
    raise SystemExit(1)
if not same_path(meta.get("notone_model_dir"), expected_model):
    raise SystemExit(1)
PY
}

echo "============================================================================"
echo "FA + STANDALONE NO-TONE MAIN DECOUPLED RESCORING"
echo "stage=${stage}, stop_stage=${stop_stage}"
echo "FA model:             ${FA_MODEL_DIR}"
echo "standalone NT model:  ${NOTONE_MODEL_DIR}"
echo "standalone NT dict:   ${NOTONE_DICT}"
echo "score: S_main_FA + alpha*S_notone_MAIN"
echo "FA auxiliary heads: not loaded or computed during TABR decoding"
echo "Dev alpha grid: ${ALPHA_GRID}"
echo "beam_size=${BEAM_SIZE}, token_prune=${TOKEN_PRUNE}"
echo "score_norm=${SCORE_NORM}"
echo "output=${OUT_ROOT}"
echo "============================================================================"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "[Stage 0] Preflight"
  for f in \
    "${DEV_LIST}" \
    "${TEST_LIST}" \
    "${MAIN_DICT}" \
    "${NOTONE_DICT}" \
    "${FA_MODEL_DIR}/config.json" \
    "${NOTONE_MODEL_DIR}/config.json" \
    "${SCRIPT_DIR}/decode_xlsr_aux_nbest.py" \
    "${SCRIPT_DIR}/attach_notone_main_to_fa_cache.py" \
    "${SCRIPT_DIR}/search_xlsr_rescore_weights.py" \
    "${SCRIPT_DIR}/apply_xlsr_rescore.py" \
    "${SCRIPT_DIR}/ctc_rescore_utils.py" \
    "${SCRIPT_DIR}/common.py"; do
    if [ ! -f "${f}" ]; then
      echo "❌ Missing required file: ${f}"
      exit 1
    fi
  done

  python - "${NOTONE_MODEL_DIR}" "${NOTONE_DICT}" <<'PY'
import sys
from pathlib import Path
from transformers import Wav2Vec2ForCTC

model_dir = Path(sys.argv[1])
dict_path = Path(sys.argv[2])
ids = []
with dict_path.open("r", encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise SystemExit(f"bad dictionary line {lineno}: {line}")
        try:
            idx = int(parts[0])
        except ValueError:
            idx = int(parts[1])
        ids.append(idx)

expected = max(ids) + 1
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
actual = int(model.config.vocab_size)
print(f"standalone no-tone vocab: dict={expected}, model={actual}")
if expected != actual:
    raise SystemExit("❌ standalone no-tone model/dictionary mismatch")
print("✅ standalone no-tone checkpoint/dictionary preflight passed")
PY
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[Stage 1] Build DEV FA + standalone-no-tone score cache"

  DEV_FA_CACHE=""
  if [ "${FORCE_CACHE}" != "1" ] \
     && [ -s "${DEV_REUSE_CACHE}" ] \
     && validate_reuse_cache "${DEV_REUSE_META}" "${FA_MODEL_DIR}" "${DEV_LIST}"
  then
    DEV_FA_CACHE="${DEV_REUSE_CACHE}"
    echo "Reuse validated FA DEV cache: ${DEV_FA_CACHE}"
  else
    DEV_FA_CACHE="${DEV_FA_LOCAL_CACHE}"
    if [ "${FORCE_CACHE}" = "1" ] || [ ! -s "${DEV_FA_CACHE}" ] \
       || ! validate_reuse_cache "${DEV_FA_CACHE%.jsonl}.meta.json" "${FA_MODEL_DIR}" "${DEV_LIST}"
    then
      echo "Generate FA DEV N-best + exact FA main CTC scores only"
      python "${SCRIPT_DIR}/decode_xlsr_aux_nbest.py" \
        --model_dir "${FA_MODEL_DIR}" \
        --data_list "${DEV_LIST}" \
        --dict_path "${MAIN_DICT}" \
        --output_jsonl "${DEV_FA_CACHE}" \
        --root "${S0_ROOT}" \
        --task fax \
        --main_ctc_only \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --beam_size "${BEAM_SIZE}" \
        --token_prune "${TOKEN_PRUNE}" \
        "${FP16_ARG[@]}"
    else
      echo "Reuse local FA DEV cache: ${DEV_FA_CACHE}"
    fi
  fi

  if [ "${FORCE_CACHE}" = "1" ] || [ ! -s "${DEV_MERGED_CACHE}" ] \
     || ! validate_merged_cache "${DEV_MERGED_CACHE%.jsonl}.meta.json" "${DEV_FA_CACHE}" "${DEV_LIST}"
  then
    python "${SCRIPT_DIR}/attach_notone_main_to_fa_cache.py" \
      --base_cache "${DEV_FA_CACHE}" \
      --notone_model_dir "${NOTONE_MODEL_DIR}" \
      --notone_dict_path "${NOTONE_DICT}" \
      --data_list "${DEV_LIST}" \
      --main_dict_path "${MAIN_DICT}" \
      --output_jsonl "${DEV_MERGED_CACHE}" \
      --root "${S0_ROOT}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      "${FP16_ARG[@]}"
  else
    echo "Reuse merged DEV cache: ${DEV_MERGED_CACHE}"
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "[Stage 2] Search alpha on DEV only (beta=0)"

  if [ ! -s "${DEV_MERGED_CACHE}" ]; then
    echo "❌ Missing DEV merged cache: ${DEV_MERGED_CACHE}"
    echo "Run stage 1 first."
    exit 1
  fi

  python "${SCRIPT_DIR}/search_xlsr_rescore_weights.py" \
    --cache "${DEV_MERGED_CACHE}" \
    --output_dir "${DEV_OUT}" \
    --task fnx \
    --alpha_grid "${ALPHA_GRID}" \
    --score_norm "${SCORE_NORM}"

  python - "${DEV_OUT}/best_weights.json" "${NOTONE_MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
j = json.loads(p.read_text(encoding="utf-8"))
j["task"] = "fa_plus_standalone_notone_main"
j["score_definition"] = "S_main_FA + alpha*S_notone_MAIN"
j["notone_model_dir"] = sys.argv[2]
j["beta_tone"] = 0.0
p.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
PY

  echo
  echo "DEV-selected weight:"
  cat "${DEV_OUT}/best_weights.json"
  echo
  echo "Full Dev curve:"
  cat "${DEV_OUT}/weight_search.tsv"

  python - "${DEV_OUT}/best_weights.json" "${ALPHA_GRID}" <<'PY'
import json
import sys

best = json.load(open(sys.argv[1], "r", encoding="utf-8"))
grid = [float(x.strip()) for x in sys.argv[2].split(",") if x.strip()]
a = float(best["alpha_notone"])
mx = max(grid)
if abs(a - mx) < 1e-12:
    print(
        f"WARNING: best alpha={a:g} is the UPPER search boundary ({mx:g}).\n"
        "Expand ALPHA_GRID on Dev before touching Test."
    )
else:
    print(f"✅ Dev optimum alpha={a:g} is not the upper boundary {mx:g}.")
PY
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "[Stage 3] Apply frozen DEV alpha to TEST exactly once"

  if [ ! -f "${DEV_OUT}/best_weights.json" ]; then
    echo "❌ Missing Dev-selected weights: ${DEV_OUT}/best_weights.json"
    echo "Run stage 2 first."
    exit 1
  fi

  if [ "${ALLOW_UPPER_BOUNDARY_TEST}" != "1" ]; then
    python - "${DEV_OUT}/best_weights.json" "${ALPHA_GRID}" <<'PY'
import json
import sys

best = json.load(open(sys.argv[1], "r", encoding="utf-8"))
grid = [float(x.strip()) for x in sys.argv[2].split(",") if x.strip()]
a = float(best["alpha_notone"])
mx = max(grid)
if abs(a - mx) < 1e-12:
    raise SystemExit(
        f"❌ Refusing Test: Dev-selected alpha={a:g} is the upper grid boundary. "
        "Expand ALPHA_GRID on Dev first. Set ALLOW_UPPER_BOUNDARY_TEST=1 only "
        "if you deliberately want to override this guard."
    )
PY
  fi

  TEST_FA_CACHE=""
  if [ "${FORCE_CACHE}" != "1" ] \
     && [ -s "${TEST_REUSE_CACHE}" ] \
     && validate_reuse_cache "${TEST_REUSE_META}" "${FA_MODEL_DIR}" "${TEST_LIST}"
  then
    TEST_FA_CACHE="${TEST_REUSE_CACHE}"
    echo "Reuse validated FA TEST cache: ${TEST_FA_CACHE}"
  else
    TEST_FA_CACHE="${TEST_FA_LOCAL_CACHE}"
    if [ "${FORCE_CACHE}" = "1" ] || [ ! -s "${TEST_FA_CACHE}" ] \
       || ! validate_reuse_cache "${TEST_FA_CACHE%.jsonl}.meta.json" "${FA_MODEL_DIR}" "${TEST_LIST}"
    then
      echo "Generate FA TEST N-best + exact FA main CTC scores only"
      python "${SCRIPT_DIR}/decode_xlsr_aux_nbest.py" \
        --model_dir "${FA_MODEL_DIR}" \
        --data_list "${TEST_LIST}" \
        --dict_path "${MAIN_DICT}" \
        --output_jsonl "${TEST_FA_CACHE}" \
        --root "${S0_ROOT}" \
        --task fax \
        --main_ctc_only \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --beam_size "${BEAM_SIZE}" \
        --token_prune "${TOKEN_PRUNE}" \
        "${FP16_ARG[@]}"
    else
      echo "Reuse local FA TEST cache: ${TEST_FA_CACHE}"
    fi
  fi

  if [ "${FORCE_CACHE}" = "1" ] || [ ! -s "${TEST_MERGED_CACHE}" ] \
     || ! validate_merged_cache "${TEST_MERGED_CACHE%.jsonl}.meta.json" "${TEST_FA_CACHE}" "${TEST_LIST}"
  then
    python "${SCRIPT_DIR}/attach_notone_main_to_fa_cache.py" \
      --base_cache "${TEST_FA_CACHE}" \
      --notone_model_dir "${NOTONE_MODEL_DIR}" \
      --notone_dict_path "${NOTONE_DICT}" \
      --data_list "${TEST_LIST}" \
      --main_dict_path "${MAIN_DICT}" \
      --output_jsonl "${TEST_MERGED_CACHE}" \
      --root "${S0_ROOT}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      "${FP16_ARG[@]}"
  else
    echo "Reuse merged TEST cache: ${TEST_MERGED_CACHE}"
  fi

  python "${SCRIPT_DIR}/apply_xlsr_rescore.py" \
    --cache "${TEST_MERGED_CACHE}" \
    --weights "${DEV_OUT}/best_weights.json" \
    --output_dir "${TEST_OUT}"

  echo
  echo "================ FA + STANDALONE NO-TONE MAIN ==================="
  echo "Score: S_main_FA + alpha*S_notone_MAIN"
  echo "FA:      ${FA_MODEL_DIR}"
  echo "No-tone: ${NOTONE_MODEL_DIR}"
  echo
  echo "DEV-selected weights:"
  cat "${DEV_OUT}/best_weights.json"
  echo
  echo "DEV:"
  cat "${DEV_OUT}/ser"
  cat "${DEV_OUT}/cer"
  echo
  echo "TEST:"
  cat "${TEST_OUT}/ser"
  cat "${TEST_OUT}/cer"
  echo "================================================================="
fi
