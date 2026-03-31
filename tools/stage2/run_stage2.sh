#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ONLY="${RUN_ONLY:-all}"          # all / ms / tirms / rgbtirms
MS_SEEDS="${MS_SEEDS:-11 17 23}"
FUSION_SEED="${FUSION_SEED:-17}"

mkdir -p logs_stage2 stage2_reports stage2_generated models_stage2 backups

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_dir="backups/stage2_${timestamp}"
mkdir -p "$backup_dir"

find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.yml" \) \
  | grep -Ev '/(\.git|venv|env|site-packages|__pycache__|stage2_generated|tools/stage2)/' \
  | grep -E '(train|run|launch|config|yaml|yml)' \
  | while read -r f; do cp --parents "$f" "$backup_dir" 2>/dev/null || true; done

mapfile -t ALL_PY < <(
  find . -type f -name "*.py" \
    | grep -Ev '/(\.git|venv|env|site-packages|__pycache__|stage2_generated|tools/stage2|backups)/'
)

pick_script() {
  local pattern f b hit
  shopt -s nocasematch
  for pattern in "$@"; do
    for f in "${ALL_PY[@]}"; do
      b="$(basename "$f")"
      if [[ "$b" =~ $pattern ]]; then
        echo "$f"
        shopt -u nocasematch
        return 0
      fi
    done
  done
  shopt -u nocasematch

  for pattern in "$@"; do
    hit="$(grep -RIlE "$pattern" . --include="*.py" | grep -Ev '/(\.git|venv|env|site-packages|__pycache__|stage2_generated|tools/stage2)/' | head -n1 || true)"
    if [[ -n "$hit" ]]; then
      echo "$hit"
      return 0
    fi
  done
  return 1
}

get_help() {
  local script="$1"
  timeout 5s python "$script" -h 2>&1 || true
}

append_first_supported() {
  local help="$1"
  local -n arr="$2"
  shift 2
  while (( $# >= 2 )); do
    local flag="$1"
    local value="$2"
    shift 2
    if grep -q -- "$flag" <<< "$help"; then
      arr+=("$flag" "$value")
      return 0
    fi
  done
  return 1
}

append_bool_supported() {
  local help="$1"
  local -n arr="$2"
  shift 2
  while (( $# >= 1 )); do
    local flag="$1"
    shift 1
    if grep -q -- "$flag" <<< "$help"; then
      arr+=("$flag")
      return 0
    fi
  done
  return 1
}

make_runnable_script() {
  local src="$1"
  local tag="$2"
  local help dst

  help="$(get_help "$src")"
  if grep -Eq -- '--epochs|--num-epochs|--max-epochs|--lr|--learning-rate|--seed|--scheduler' <<< "$help"; then
    echo "$src"
    return 0
  fi

  dst="stage2_generated/$(basename "${src%.py}")_${tag}_stage2.py"
  python tools/stage2/patch_training_script.py "$src" "$dst" "$tag" | tee "stage2_reports/patch_${tag}_${timestamp}.txt" >&2
  echo "$dst"
}

run_tag() {
  local tag="$1"
  local src="$2"
  local seed="${3:-}"
  local log="logs_stage2/${tag}_${timestamp}.log"

  if [[ -z "${src:-}" ]]; then
    echo "[WARN] skip ${tag}: script not found" | tee -a "$log"
    return 0
  fi

  local script help
  script="$(make_runnable_script "$src" "$tag")"
  help="$(get_help "$script")"

  local -a cmd=(python "$script")

  append_first_supported "$help" cmd --epochs 100 --num-epochs 100 --max-epochs 100 || true
  append_first_supported "$help" cmd --lr 3e-4 --learning-rate 3e-4 || true
  append_first_supported "$help" cmd --weight-decay 1e-4 --wd 1e-4 || true
  append_first_supported "$help" cmd --patience 15 --early-stop-patience 15 || true
  append_first_supported "$help" cmd --label-smoothing 0.05 || true
  append_first_supported "$help" cmd --save-dir "models_stage2/${tag}" --model-dir "models_stage2/${tag}" --output-dir "models_stage2/${tag}" || true
  append_first_supported "$help" cmd --log-dir "logs_stage2" || true
  append_first_supported "$help" cmd --device cuda || true

  if [[ -n "$seed" ]]; then
    append_first_supported "$help" cmd --seed "$seed" --random-seed "$seed" || true
  fi

  if grep -q -- "--scheduler" <<< "$help"; then
    cmd+=(--scheduler cosine)
  else
    append_bool_supported "$help" cmd --cosine --use-cosine || true
  fi

  echo "[RUN] tag=${tag}" | tee "$log"
  echo "[RUN] src=${src}" | tee -a "$log"
  echo "[RUN] script=${script}" | tee -a "$log"
  echo "[RUN] cmd=${cmd[*]}" | tee -a "$log"
  "${cmd[@]}" 2>&1 | tee -a "$log"
}

MS_SRC="$(pick_script 'ms_single|single_ms|(train|run).*(ms.*single|single.*ms)' || true)"
TIR_MS_SRC="$(pick_script 'tir_ms|ms_tir|(train|run).*(tir.*ms|ms.*tir)' || true)"
RGB_TIR_MS_SRC="$(pick_script 'rgb_tir_ms|rgb_ms_tir|tir_rgb_ms|tir_ms_rgb|ms_rgb_tir|ms_tir_rgb|(train|run).*(rgb.*tir.*ms|rgb.*ms.*tir|tir.*rgb.*ms|tir.*ms.*rgb|ms.*rgb.*tir|ms.*tir.*rgb)' || true)"

{
  echo "timestamp: ${timestamp}"
  echo "RUN_ONLY: ${RUN_ONLY}"
  echo "MS_SRC: ${MS_SRC:-NOT_FOUND}"
  echo "TIR_MS_SRC: ${TIR_MS_SRC:-NOT_FOUND}"
  echo "RGB_TIR_MS_SRC: ${RGB_TIR_MS_SRC:-NOT_FOUND}"
} | tee "stage2_reports/discovery_${timestamp}.txt"

case "$RUN_ONLY" in
  all|ms)
    for s in $MS_SEEDS; do
      run_tag "ms_seed${s}" "${MS_SRC:-}" "$s"
    done
    ;;
esac

case "$RUN_ONLY" in
  all|tirms)
    run_tag "tirms_seed${FUSION_SEED}" "${TIR_MS_SRC:-}" "${FUSION_SEED}"
    ;;
esac

case "$RUN_ONLY" in
  all|rgbtirms)
    run_tag "rgbtirms_seed${FUSION_SEED}" "${RGB_TIR_MS_SRC:-}" "${FUSION_SEED}"
    ;;
esac

python tools/stage2/parse_logs.py "logs_stage2/*_${timestamp}.log" | tee "stage2_reports/summary_${timestamp}.txt"
echo "[DONE] summary=stage2_reports/summary_${timestamp}.txt"
