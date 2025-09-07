#!/usr/bin/env bash
set -euo pipefail
python evaluation/eval_pipeline.py --config configs/default.yaml "$@" 