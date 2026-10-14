#!/usr/bin/env bash
# Smoke test for the vision-LLM experiments (V1/V2/V3 + metrics).
#
# Validates the whole pipeline end-to-end on a tiny subset before the real
# background run is launched:
#   1. Preflight: Ollama server reachable, both vision models installed,
#      test images present, pycocoevalcap venv present.
#   2. Runs run_vision.sh --limit N --no-meteor (caption en/ar + judge +
#      metrics for every model).
#   3. Verifies the generated result files are sane.
#   4. Cleans up the smoke-generated result files so the real run starts from
#      a clean analysis/results.
#
# Exits 0 on success, non-zero on any failure.
#
# Usage:  ./smoke_vision.sh [--limit N]   (default limit: 2)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
XLSX_PATH="$REPO_ROOT/data/arabic_captions.xlsx"
IMAGE_DIR="$REPO_ROOT/data/coco/train2014"
RESULTS_DIR="$REPO_ROOT/analysis/results"
LOG_DIR="$REPO_ROOT/llm_improvements/outputs"
VENV_PY="${VENV_PY:-$HOME/venvs/capcaption/bin/python}"
if [ -n "${OLLAMA_HOST:-}" ]; then
    HOST="$OLLAMA_HOST"
    case "$HOST" in
        0.0.0.0*) HOST="http://localhost:11434" ;;
        http*) : ;;
        *) HOST="http://$HOST" ;;
    esac
else
    HOST="http://localhost:11434"
fi

MODELS=("gemma4:12b" "qwen3.8:27b")
LIMIT=2

while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

fail() { echo "[$(date '+%F %T')] SMOKE FAIL: $1"; exit 1; }

echo "[$(date '+%F %T')] === vision smoke test (limit=$LIMIT) ==="

# --- 1. Preflight ------------------------------------------------------------
python3 - "$HOST" "${MODELS[@]}" <<'PY' || fail "ollama preflight"
import json
import sys
import urllib.request

host, wants = sys.argv[1], sys.argv[2:]
try:
    data = json.load(urllib.request.urlopen(host + "/api/tags", timeout=30))
except Exception as e:
    raise SystemExit(f"cannot reach ollama at {host}: {e}")
have = {m["name"] for m in data.get("models", [])}
missing = [m for m in wants if m not in have]
if missing:
    raise SystemExit(f"missing vision models: {', '.join(missing)}")
print(f"ollama OK at {host} -> models: {', '.join(wants)}")
PY
[ -d "$IMAGE_DIR" ] && [ -n "$(ls -A "$IMAGE_DIR" 2>/dev/null)" ] \
    || fail "no test images in $IMAGE_DIR"
[ -f "$XLSX_PATH" ] || fail "workbook missing: $XLSX_PATH"
[ -x "$VENV_PY" ] || fail "metrics venv missing: $VENV_PY"
"$VENV_PY" -c "import pycocoevalcap" >/dev/null 2>&1 \
    || fail "pycocoevalcap not importable in $VENV_PY"

# --- 2. Run the pipeline on a tiny subset ------------------------------------
mkdir -p "$LOG_DIR"
echo "[$(date '+%F %T')] running run_vision.sh (limit=$LIMIT, --no-meteor)..."
"$SCRIPT_DIR/run_vision.sh" --limit "$LIMIT" --no-meteor \
    >>"$LOG_DIR/vision_smoke.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] run_vision.sh returned $rc"
if [ "$rc" -ne 0 ]; then
    tail -60 "$LOG_DIR/vision_smoke.log"
    fail "run_vision.sh failed (rc=$rc); see $LOG_DIR/vision_smoke.log"
fi

# --- 3. Validate generated results -------------------------------------------
python3 - "$RESULTS_DIR" "${MODELS[@]}" <<'PY' || fail "result validation"
import json
import os
import sys

results_dir, models = sys.argv[1], sys.argv[2:]
slug = lambda m: "".join(c if c.isalnum() else "_" for c in m).strip("_")
ok = True
for m in models:
    s = slug(m)
    for lang in ("en", "ar"):
        jp = os.path.join(results_dir, f"vlm_caption__{s}__{lang}.json")
        if not os.path.exists(jp):
            print(f"  MISSING {jp}"); ok = False; continue
        d = json.load(open(jp))
        per = d.get("per_item") or []
        n_ok = sum(1 for x in per if x.get("caption"))
        print(f"  caption {lang} [{m}]: n={len(per)} captioned={n_ok}")
        if len(per) < 1 or n_ok < 1:
            ok = False
    for lang, ref in (("en", "eng_refs"), ("ar", "ar_google"), ("ar", "ar_manual")):
        mjp = os.path.join(results_dir,
                           f"vlm_caption__{s}__{lang}__metrics__{ref}.json")
        if not os.path.exists(mjp):
            print(f"  MISSING {mjp}"); ok = False; continue
        d = json.load(open(mjp))
        metrics = d.get("metrics") or {}
        bleu1 = metrics.get("BLEU-1")
        n = d.get("n", 0)
        if ref == "ar_manual":
            ok_metric = ((n == 0 and bleu1 is None)
                         or isinstance(bleu1, float))
        else:
            ok_metric = isinstance(bleu1, float)
        print(f"  metrics {ref} [{m}]: n={n} BLEU-1={bleu1}")
        if not ok_metric:
            print(f"  BAD metrics {mjp}: unexpected BLEU-1 value {bleu1!r}")
            ok = False
    jp = os.path.join(results_dir, f"vlm_judge__{s}.json")
    if not os.path.exists(jp):
        print(f"  MISSING {jp}"); ok = False; continue
    d = json.load(open(jp))
    scored = (d.get("summary") or {}).get("n", 0)
    print(f"  judge [{m}]: judged={d.get('n')} scored={scored} failures={d.get('failures')}")
    if scored < 1:
        ok = False
if not ok:
    raise SystemExit("one or more smoke outputs are invalid")
print("result validation OK")
PY
[ $? -eq 0 ] || exit 1

# --- 4. Clean up smoke-generated results --------------------------------------
python3 - "$RESULTS_DIR" "${MODELS[@]}" <<'PY' || exit 1
import glob
import os
import sys

results_dir, models = sys.argv[1], sys.argv[2:]
slug = lambda m: "".join(c if c.isalnum() else "_" for c in m).strip("_")
removed = []
for m in models:
    s = slug(m)
    for pat in (
        f"vlm_caption__{s}__*.json", f"vlm_caption__{s}__*.txt",
        f"vlm_caption__{s}__*__metrics__*.json",
        f"vlm_judge__{s}.*",
    ):
        for p in glob.glob(os.path.join(results_dir, pat)):
            os.remove(p)
            removed.append(os.path.basename(p))
print(f"removed {len(removed)} smoke result file(s)")
PY

echo "[$(date '+%F %T')] SMOKE PASS"