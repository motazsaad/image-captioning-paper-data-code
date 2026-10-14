#!/usr/bin/env python3
"""Prepare and optionally score reference-style sensitivity comparisons.

The script consumes a crosslingual_consistency JSON.  It writes three
vlm_metrics.py-compatible caption files: direct English VLM captions against
English references, original Arabic VLM captions against Google-MT Arabic
references, and back-translated Arabic VLM captions against English references.
Pass --metrics-python to run the official COCO metrics on all three files.
"""
import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "analysis", "scripts")
RESULTS_DIR = os.path.join(REPO_ROOT, "analysis", "results")


def write(path, model, lang, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "lang": lang, "n": len(records), "per_item": records},
                  f, ensure_ascii=False, indent=2)


def run_metrics(python, path, refs, no_meteor):
    cmd = [python, os.path.join(SCRIPTS_DIR, "vlm_metrics.py"), "--caption-json", path, "--refs", refs]
    if no_meteor:
        cmd.append("--no-meteor")
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return path.replace(".json", f"__metrics__{refs}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crosslingual-json", required=True)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--metrics-python", default=None,
                    help="Python executable with pycocoevalcap installed")
    ap.add_argument("--no-meteor", action="store_true")
    args = ap.parse_args()
    with open(args.crosslingual_json, encoding="utf-8") as f:
        source = json.load(f)
    tag = os.path.basename(args.crosslingual_json).replace("crosslingual_consistency__", "").replace(".json", "")
    os.makedirs(args.out_dir, exist_ok=True)
    direct_en, original_ar, back_en = [], [], []
    for item in source.get("per_item", []):
        if item.get("error"):
            continue
        common = {"key": item.get("key"), "eng_refs": item.get("eng_refs", []),
                  "ar_google": item.get("ar_google", []), "ar_manual": []}
        if item.get("english_caption") and common["eng_refs"]:
            direct_en.append({**common, "caption": item["english_caption"]})
        if item.get("arabic_caption") and common["ar_google"]:
            original_ar.append({**common, "caption": item["arabic_caption"]})
        if item.get("backtranslated_en") and common["eng_refs"]:
            back_en.append({**common, "caption": item["backtranslated_en"]})
    paths = {
        "direct_english": os.path.join(args.out_dir, f"reference_style__{tag}__direct_en.json"),
        "arabic_google_reference": os.path.join(args.out_dir, f"reference_style__{tag}__arabic_google.json"),
        "backtranslated_english": os.path.join(args.out_dir, f"reference_style__{tag}__backtranslated_en.json"),
    }
    write(paths["direct_english"], source.get("english_model"), "en", direct_en)
    write(paths["arabic_google_reference"], source.get("arabic_model"), "ar", original_ar)
    write(paths["backtranslated_english"], source.get("arabic_model") + " backtranslated", "en", back_en)
    result = {"experiment": "reference-style sensitivity", "inputs": args.crosslingual_json,
              "caption_files": paths, "n": {"direct_english": len(direct_en),
              "arabic_google_reference": len(original_ar), "backtranslated_english": len(back_en)}}
    if args.metrics_python:
        result["metric_files"] = {
            "direct_english": run_metrics(args.metrics_python, paths["direct_english"], "eng_refs", args.no_meteor),
            "arabic_google_reference": run_metrics(args.metrics_python, paths["arabic_google_reference"], "ar_google", args.no_meteor),
            "backtranslated_english": run_metrics(args.metrics_python, paths["backtranslated_english"], "eng_refs", args.no_meteor),
        }
    report = os.path.join(args.out_dir, f"reference_style__{tag}__manifest.json")
    with open(report, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved -> {report}")


if __name__ == "__main__":
    main()
