#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# PnP-OVSS launcher
# Usage:
#   ./run.sh                          # infer with config.yaml
#   ./run.sh infer                    # same
#   ./run.sh tune                     # grid search with config.yaml
#   ./run.sh infer my_config.yaml     # custom config
#   ./run.sh tune  my_config.yaml
#   ./run.sh infer config.yaml --layer 5 --head 3   # extra CLI overrides
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
MODE="eval"
CONFIG="config.yaml"

# Parse positional args (mode and/or config path)
for arg in "$@"; do
  case "$arg" in
    infer|tune|eval) MODE="$arg" ;;
    *.yaml|*.yml) CONFIG="$arg" ;;
  esac
done

# Collect any extra flags that aren't mode/config (pass through to python)
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    infer|tune|*.yaml|*.yml) ;;
    *) EXTRA+=("$arg") ;;
  esac
done

# ── Resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Pick python ───────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"

echo "========================================"
echo "  PnP-OVSS"
echo "  mode   : $MODE"
echo "  config : $CONFIG"
[ ${#EXTRA[@]} -gt 0 ] && echo "  extra  : ${EXTRA[*]}"
echo "========================================"
echo ""

case "$MODE" in
  infer)
    "$PYTHON" main.py --config "$CONFIG" "${EXTRA[@]}"
    ;;
  tune)
    "$PYTHON" scripts/tune_hyperparams.py --config "$CONFIG" "${EXTRA[@]}"
    ;;
  eval)
    "$PYTHON" scripts/evaluate.py --config "$CONFIG" "${EXTRA[@]}"
    ;;
  *)
    echo "Unknown mode '$MODE'. Use 'infer', 'tune', or 'eval'." >&2
    exit 1
    ;;
esac
