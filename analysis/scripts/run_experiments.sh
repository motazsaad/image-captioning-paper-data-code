#!/usr/bin/env bash
# Run the three LLM-based experiments across the additional models, keeping the
# existing qwen3:8b / all-minilm results untouched.
#
# Experiments (each writes per-model tagged files under analysis/results):
#   1. LLM-as-Judge          (--model)                       -> 3 runs
#   2. LLM error analysis    (--model)                       -> 3 runs
#   3. Round-trip back-trans (--model x --embed-model)       -> 6 runs
#
# Usage:  ./run_experiments.sh [--limit N] [--dry-run]
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
XLSX_PATH="$REPO_ROOT/data/arabic_captions.xlsx"
LOG_DIR="$REPO_ROOT/llm_improvements/outputs"

LLM_MODELS=("gemma4:12b" "qwen3.8:27b" "ministral-3:14b")
EMBED_MODELS=("bge-m3:latest" "qwen3-embedding:4b")

LIMIT=0
DRY_RUN=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LIMIT_ARG=""
[ "$LIMIT" -gt 0 ] && LIMIT_ARG="--limit $LIMIT"

mkdir -p "$LOG_DIR"

run() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    echo "[$(date '+%F %T')] START $name -> $log"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  (dry-run) python3 $*"
        return
    fi
    if python3 "$@" >"$log" 2>&1; then
        echo "[$(date '+%F %T')] DONE  $name"
    else
        echo "[$(date '+%F %T')] FAIL  $name (see $log)"
    fi
}

for M in "${LLM_MODELS[@]}"; do
    run "judge__$(echo "$M" | tr ':' '_')" \
        "$SCRIPT_DIR/llm_judge.py" --model "$M" --xlsx --xlsx-path "$XLSX_PATH" $LIMIT_ARG
    run "error__$(echo "$M" | tr ':' '_')" \
        "$SCRIPT_DIR/llm_error_analysis.py" --model "$M" --xlsx --xlsx-path "$XLSX_PATH" $LIMIT_ARG
done

for M in "${LLM_MODELS[@]}"; do
    for E in "${EMBED_MODELS[@]}"; do
        run "backtranslate__$(echo "$M" | tr ':' '_')__$(echo "$E" | tr ':' '_')" \
            "$SCRIPT_DIR/llm_backtranslate.py" --model "$M" --embed-model "$E" \
            --xlsx --xlsx-path "$XLSX_PATH" $LIMIT_ARG
    done
done

echo "[$(date '+%F %T')] ALL DONE"