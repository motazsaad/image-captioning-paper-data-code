#!/usr/bin/env bash
# Arabic prompt ablation for the two zero-shot VLMs. The existing Arabic run is
# the COCO-style baseline; this script adds concise and grounded variants and
# evaluates all new outputs against identical Google-MT Arabic references.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_DIR="${IMAGE_DIR:-$REPO_ROOT/data/coco/train2014}"
XLSX_PATH="${XLSX_PATH:-$REPO_ROOT/data/arabic_captions.xlsx}"
RESULTS_DIR="$REPO_ROOT/analysis/results"
LIMIT=0
NO_METEOR=0
DRY_RUN=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --no-meteor) NO_METEOR=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
python3 -c "import requests, openpyxl, pycocoevalcap" >/dev/null || {
    echo "Activate the caption-metrics environment first:" >&2
    echo "  source .venv-caption-metrics/bin/activate" >&2
    exit 1
}
[ -d "$IMAGE_DIR" ] || { echo "Missing images: $IMAGE_DIR" >&2; exit 1; }
LIMIT_ARGS=()
[ "$LIMIT" -gt 0 ] && LIMIT_ARGS=(--limit "$LIMIT")
METEOR_ARGS=()
[ "$NO_METEOR" -eq 1 ] && METEOR_ARGS=(--no-meteor)

# Fixed prompts are deliberately short so the experiment isolates instruction
# style rather than a long chain of task-specific heuristics.
declare -A PROMPTS
PROMPTS[concise]='صف محتوى هذه الصورة في جملة عربية فصيحة واحدة قصيرة، لا تزيد عن 12 كلمة. أخرج الوصف فقط دون أي تعليق أو ترجمة.'
PROMPTS[grounded]='صف فقط ما هو واضح ومرئي في هذه الصورة في جملة عربية فصيحة واحدة. تجنب التخمين والتفاصيل غير المؤكدة ولا تذكر كائنات أو علاقات لا تستطيع رؤيتها بوضوح. أخرج الوصف فقط.'

run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'DRY RUN: '; printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}
slug() { printf '%s' "$1" | tr -c 'A-Za-z0-9' '_'; }

for model in "qwen3.8:27b" "gemma4:12b"; do
    model_tag="$(slug "$model")"
    for condition in concise grounded; do
        out="$RESULTS_DIR/vlm_caption__${model_tag}__ar__${condition}.json"
        run python3 "$SCRIPT_DIR/vlm_caption.py" --model "$model" --lang ar \
            --run-tag "$condition" --prompt "${PROMPTS[$condition]}" \
            --image-dir "$IMAGE_DIR" --xlsx "$XLSX_PATH" "${LIMIT_ARGS[@]}"
        run python3 "$SCRIPT_DIR/vlm_metrics.py" --caption-json "$out" \
            --refs ar_google "${METEOR_ARGS[@]}"
    done
done
