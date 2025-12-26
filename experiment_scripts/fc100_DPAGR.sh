#!/usr/bin/env bash
set -euo pipefail

python DPAGR.py \
  --name_of_args_json_file experiment_config/fc100_5_8_0.01_48_5_0.json \
  --gpu_to_use 0

