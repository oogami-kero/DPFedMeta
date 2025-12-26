#!/usr/bin/env bash
set -euo pipefail

GPU_TO_USE="${1:-0}"
SKIP_NODP="${SKIP_NODP:-0}"
EPS_LIST="${EPS_LIST:-"1 2 4 8 16"}"

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

if [[ "${SKIP_NODP}" != "1" ]]; then
  run "miniImageNet IID 5-way 5-shot (no-DP)" \
    python train_maml_system.py \
      --gpu_to_use "${GPU_TO_USE}" \
      --continue_from_epoch from_scratch \
      --name_of_args_json_file experiment_config/mini-imagenet_1_2_0.01_48_5_0_iid_5shot_nodp_seed890454.json
fi

for eps in ${EPS_LIST}; do
  run "miniImageNet IID 5-way 5-shot (DP-AGR, eps=${eps}, delta=1e-5)" \
    python DPAGR.py \
      --gpu_to_use "${GPU_TO_USE}" \
      --continue_from_epoch from_scratch \
      --name_of_args_json_file "experiment_config/mini-imagenet_1_2_0.01_48_5_0_iid_5shot_eps${eps}_seed890454.json"
done
