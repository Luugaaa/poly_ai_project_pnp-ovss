#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# PnP-OVSS launcher
# Usage:
#   ./run.sh                          # eval with default config
#   ./run.sh eval                     # same
#   ./run.sh infer                    # run single inference flow
#   ./run.sh tune                     # layer/head grid search
#   ./run.sh tune_pipeline            # pipeline param search (dropout, patches, granularity)
#   ./run.sh infer my_config.yaml     # custom config
#   ./run.sh tune  my_config.yaml
#   ./run.sh tune_pipeline my_config.yaml
#   ./run.sh eval config.yaml --max_images 50        # extra CLI overrides
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODE="eval"
CONFIG="config_regular_tune.yaml"

# Parse positional args (mode and/or config path)
for arg in "$@"; do
  case "$arg" in
    infer|tune|tune_pipeline|eval) MODE="$arg" ;;
    *.yaml|*.yml) CONFIG="$arg" ;;
  esac
done

# Collect any extra flags that aren't mode/config (pass through to python)
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    infer|tune|tune_pipeline|eval|*.yaml|*.yml) ;;
    *) EXTRA+=("$arg") ;;
  esac
done

# ── Resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Pick python ───────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-.venv/bin/python}"

echo "========================================"
echo "  PnP-OVSS"
echo "  mode   : $MODE"
echo "  config : $CONFIG"
[ ${#EXTRA[@]} -gt 0 ] && echo "  extra  : ${EXTRA[*]}"
echo "========================================"
echo ""

if [ $# -eq 0 ]; then
  echo "Note: default mode is eval."
  echo "Use './run.sh tune_pipeline <config.yaml>' to run tuning."
  echo ""
fi

case "$MODE" in
  infer)
    "$PYTHON" main.py --config "$CONFIG" "${EXTRA[@]}"
    ;;
  tune)
    "$PYTHON" scripts/tune_hyperparams.py --config "$CONFIG" "${EXTRA[@]}"
    ;;
  tune_pipeline)
    "$PYTHON" scripts/tune_pipeline.py --config "$CONFIG" "${EXTRA[@]}"
    ;;
  eval)
    "$PYTHON" scripts/evaluate.py --config "$CONFIG" "${EXTRA[@]}"
    ;;
  *)
    echo "Unknown mode '$MODE'. Use 'infer', 'tune', 'tune_pipeline', or 'eval'." >&2
    exit 1
    ;;
esac
