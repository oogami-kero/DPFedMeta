#!/usr/bin/env bash
set -euo pipefail

GPU_TO_USE="${1:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export DATASET_DIR="datasets/"
export PYTHONUNBUFFERED=1

run() {
  local label="$1"
  shift
  echo "[$(date '+%F %T')] ${label}"
  "$@"
  echo "[$(date '+%F %T')] DONE: ${label}"
  echo
}

run "FC100 non-IID (pseudo) 5-way 5-shot (no-DP)" \
  python train_maml_system.py \
    --gpu_to_use "${GPU_TO_USE}" \
    --continue_from_epoch from_scratch \
    --name_of_args_json_file experiment_config/fc100_5_8_0.01_48_5_0_noniid_nodp_seed890454.json

for eps in 1 2 4 8 16; do
  run "FC100 non-IID (pseudo) 5-way 5-shot (DP-AGR, eps=${eps}, delta=1e-5)" \
    python DPAGR.py \
      --gpu_to_use "${GPU_TO_USE}" \
      --continue_from_epoch from_scratch \
      --name_of_args_json_file "experiment_config/fc100_5_8_0.01_48_5_0_noniid_eps${eps}_seed890454.json"
done

