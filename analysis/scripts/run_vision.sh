#!/usr/bin/env bash
# Vision-LLM experiments (V1/V2/V3) using Ollama multimodal models.
#
#   V1  zero-shot English captioning            -> vlm_caption.py --lang en
#   V2  zero-shot Arabic captioning             -> vlm_caption.py --lang ar
#   V3  image-grounded judge of Google-MT caps  -> vlm_judge.py
#   Metrics  BLEU/METEOR/CIDEr/ROUGE-L (pycocoevalcap venv) -> vlm_metrics.py
#
# Models: gemma4:12b, qwen3.8:27b   (both vision-capable)
# Images: data/coco/train2014/ (downloaded from images.cocodataset.org)
#
# Usage:  ./run_vision.sh [--limit N] [--no-meteor] [--dry-run]
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
XLSX_PATH="$REPO_ROOT/data/arabic_captions.xlsx"
IMAGE_DIR="$REPO_ROOT/data/coco/train2014"
LOG_DIR="$REPO_ROOT/llm_improvements/outputs"
VENV_PY="${VENV_PY:-$HOME/venvs/capcaption/bin/python}"

MODELS=("gemma4:12b" "qwen3.8:27b")

LIMIT=0
METEOR_FLAG=""
DRY_RUN=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --no-meteor) METEOR_FLAG="--no-meteor"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LIMIT_ARG=""
[ "$LIMIT" -gt 0 ] && LIMIT_ARG="--limit $LIMIT"
mkdir -p "$LOG_DIR"

# --- Download images if needed -----------------------------------------------
if [ ! -d "$IMAGE_DIR" ] || [ -z "$(ls -A "$IMAGE_DIR" 2>/dev/null)" ]; then
    echo "[$(date '+%F %T')] Downloading COCO train2014 test images..."
    python3 "$SCRIPT_DIR/download_coco.py" --xlsx "$XLSX_PATH" --out-dir "$IMAGE_DIR" $LIMIT_ARG \
        >>"$LOG_DIR/download_coco.log" 2>&1 || {
        echo "ERROR: image download failed (see $LOG_DIR/download_coco.log)"; exit 1; }
    echo "[$(date '+%F %T')] Images ready: $(ls "$IMAGE_DIR" | wc -l) files"
else
    echo "[$(date '+%F %T')] Images already present: $(ls "$IMAGE_DIR" | wc -l) files"
fi

run() {
    local name="$1" py="$2"; shift 2
    local log="$LOG_DIR/vision_${name}.log"
    echo "[$(date '+%F %T')] START vision: $name -> $log"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  (dry-run) $py $*"
        return
    fi
    if "$py" "$@" >"$log" 2>&1; then
        echo "[$(date '+%F %T')] DONE  vision: $name"
    else
        echo "[$(date '+%F %T')] FAIL  vision: $name (see $log)"
    fi
}

slug() { printf '%s' "$1" | tr -c "A-Za-z0-9" "_"; }

for M in "${MODELS[@]}"; do
    # V1 / V2 zero-shot captioning (system python: requests/openpyxl only)
    run "caption__$(slug "$M")__en" python3 \
        "$SCRIPT_DIR/vlm_caption.py" --model "$M" --lang en \
        --image-dir "$IMAGE_DIR" --xlsx "$XLSX_PATH" $LIMIT_ARG
    run "caption__$(slug "$M")__ar" python3 \
        "$SCRIPT_DIR/vlm_caption.py" --model "$M" --lang ar \
        --image-dir "$IMAGE_DIR" --xlsx "$XLSX_PATH" $LIMIT_ARG
    # V3 image-grounded judge
    run "judge__$(slug "$M")" python3 \
        "$SCRIPT_DIR/vlm_judge.py" --model "$M" \
        --image-dir "$IMAGE_DIR" --xlsx "$XLSX_PATH" $LIMIT_ARG
done

# --- Metrics (venv python: pycocoevalcap) ------------------------------------
for M in "${MODELS[@]}"; do
    M_SLUG="$(slug "$M")"
    EN_JSON="$REPO_ROOT/analysis/results/vlm_caption__${M_SLUG}__en.json"
    AR_JSON="$REPO_ROOT/analysis/results/vlm_caption__${M_SLUG}__ar.json"
    run "metrics__${M_SLUG}__en__eng_refs" "$VENV_PY" \
        "$SCRIPT_DIR/vlm_metrics.py" --caption-json "$EN_JSON" \
        --refs eng_refs $METEOR_FLAG
    run "metrics__${M_SLUG}__ar__ar_google" "$VENV_PY" \
        "$SCRIPT_DIR/vlm_metrics.py" --caption-json "$AR_JSON" \
        --refs ar_google $METEOR_FLAG
    run "metrics__${M_SLUG}__ar__ar_manual" "$VENV_PY" \
        "$SCRIPT_DIR/vlm_metrics.py" --caption-json "$AR_JSON" \
        --refs ar_manual $METEOR_FLAG
done

echo "[$(date '+%F %T')] VISION ALL DONE"