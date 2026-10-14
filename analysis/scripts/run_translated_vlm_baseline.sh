#!/usr/bin/env bash
# Run direct-English-VLM -> Arabic translation baselines and score them against
# the same Google-MT Arabic references used for direct Arabic VLM captions.
# Requires an active environment containing requests and pycocoevalcap.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$REPO_ROOT/analysis/results"
TRANSLATOR="${TRANSLATOR:-ministral-3:14b}"
LIMIT=0
NO_METEOR=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --no-meteor) NO_METEOR=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
python3 -c "import requests, pycocoevalcap" >/dev/null || {
    echo "Activate the caption-metrics environment first:" >&2
    echo "  source .venv-caption-metrics/bin/activate" >&2
    exit 1
}
slug() { printf '%s' "$1" | tr -c 'A-Za-z0-9' '_'; }
LIMIT_ARGS=()
[ "$LIMIT" -gt 0 ] && LIMIT_ARGS=(--limit "$LIMIT")
METEOR_ARGS=()
[ "$NO_METEOR" -eq 1 ] && METEOR_ARGS=(--no-meteor)

for model in "qwen3.8:27b" "gemma4:12b"; do
    tag="$(slug "$model")"
    en="$RESULTS_DIR/vlm_caption__${tag}__en.json"
    out="$RESULTS_DIR/vlm_caption__${tag}__en_translated_ar.json"
    [ -f "$en" ] || { echo "Missing $en; run the English VLM captions first." >&2; exit 1; }
    python3 "$SCRIPT_DIR/translate_vlm_output.py" --english-json "$en" \
        --translator "$TRANSLATOR" "${LIMIT_ARGS[@]}"
    python3 "$SCRIPT_DIR/vlm_metrics.py" --caption-json "$out" --refs ar_google "${METEOR_ARGS[@]}"
done
