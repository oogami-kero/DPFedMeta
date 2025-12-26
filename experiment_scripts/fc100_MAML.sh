#!/usr/bin/env bash
set -euo pipefail

python train_maml_system.py \
  --name_of_args_json_file experiment_config/fc100_5_8_0.01_48_5_0_nonprivate.json \
  --gpu_to_use 0

