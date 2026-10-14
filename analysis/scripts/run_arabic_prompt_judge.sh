#!/usr/bin/env bash
# Cross-model, image-grounded evaluation of standard/concise/grounded Arabic
# caption prompts. Qwen outputs are judged by Gemma; Gemma outputs by Qwen.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$REPO_ROOT/analysis/results"
IMAGE_DIR="${IMAGE_DIR:-$REPO_ROOT/data/coco/train2014}"
LIMIT=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
python3 -c "import requests" >/dev/null || {
    echo "Activate the caption environment first: source .venv-caption-metrics/bin/activate" >&2
    exit 1
}
[ -d "$IMAGE_DIR" ] || { echo "Missing images: $IMAGE_DIR" >&2; exit 1; }
LIMIT_ARGS=()
[ "$LIMIT" -gt 0 ] && LIMIT_ARGS=(--limit "$LIMIT")
for captioner in "qwen3.8:27b" "gemma4:12b"; do
    if [ "$captioner" = "qwen3.8:27b" ]; then judge="gemma4:12b"; tag="qwen3_8_27b";
    else judge="qwen3.8:27b"; tag="gemma4_12b"; fi
    for condition in standard concise grounded; do
        if [ "$condition" = standard ]; then file="$RESULTS_DIR/vlm_caption__${tag}__ar.json";
        else file="$RESULTS_DIR/vlm_caption__${tag}__ar__${condition}.json"; fi
        [ -f "$file" ] || { echo "Missing $file" >&2; exit 1; }
        python3 "$SCRIPT_DIR/vlm_caption_judge.py" --caption-json "$file" \
            --judge-model "$judge" --image-dir "$IMAGE_DIR" "${LIMIT_ARGS[@]}"
    done
done
