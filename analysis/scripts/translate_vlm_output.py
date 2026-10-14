#!/usr/bin/env python3
"""Translate English VLM captions to Arabic for a post-translation baseline.

This creates the modern counterpart of the historical Exp3: an English
captioner produces a caption, which is subsequently translated to Arabic. The
result JSON is compatible with vlm_metrics.py and is checkpointed after each
caption so interrupted workstation jobs resume safely.
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_helper import generate, normalize_host, slug_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "analysis", "results")
PROMPT = """Translate this English image caption into one fluent Modern Standard Arabic
image caption. Preserve the described objects, actions, counts, and relations.
Output only the Arabic caption, with no explanation:

{caption}"""


def atomic_write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def payload(model, translator, records):
    return {"experiment": "English VLM output translated to Arabic", "model": model,
            "translator": translator, "lang": "ar", "n": len(records),
            "per_item": records}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--english-json", required=True, help="English VLM caption JSON")
    ap.add_argument("--translator", default="ministral-3:14b")
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    with open(args.english_json, encoding="utf-8") as f:
        source = json.load(f)
    items = [x for x in source.get("per_item", []) if x.get("key") and x.get("caption")]
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise SystemExit("No English captions found.")
    tag = slug_model(source.get("model", "vlm"))
    out_path = args.out or os.path.join(
        RESULTS_DIR, f"vlm_caption__{tag}__en_translated_ar.json")
    old = {}
    if os.path.exists(out_path) and not args.restart:
        with open(out_path, encoding="utf-8") as f:
            old = {str(x.get("key")): x for x in json.load(f).get("per_item", [])}
    host = normalize_host(args.host)
    records = [old[str(x["key"])] for x in items if str(x["key"]) in old]
    for index, item in enumerate(items, 1):
        key = str(item["key"])
        if key in old:
            continue
        rec = {"key": key, "caption": None, "source_english_caption": item["caption"],
               "eng_refs": item.get("eng_refs", []), "ar_google": item.get("ar_google", []),
               "ar_manual": item.get("ar_manual", []), "error": None}
        try:
            rec["caption"] = generate(PROMPT.format(caption=item["caption"]),
                                      model=args.translator, host=host, num_predict=160)
        except Exception as exc:
            rec["error"] = str(exc)
        records.append(rec)
        atomic_write(out_path, payload(source.get("model"), args.translator, records))
        print(f"[{index}/{len(items)}] {key}", flush=True)
    complete = payload(source.get("model"), args.translator, records)
    complete["successful"] = sum(bool(x.get("caption")) for x in records)
    complete["failures"] = sum(bool(x.get("error")) for x in records)
    atomic_write(out_path, complete)
    print(f"successful={complete['successful']} failures={complete['failures']}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
