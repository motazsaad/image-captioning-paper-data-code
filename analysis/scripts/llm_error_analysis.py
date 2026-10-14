#!/usr/bin/env python3
"""Suggestion 2 - LLM-Ranked Error / Qualitative Analysis.

The paper's Discussion lacks concrete, verified error analysis. This script
uses a local LLM to categorise the failures of the *already-generated* Arabic
captions into explainable error types:

  1. hallucination      - object/attribute mentioned but unsupported by refs
  2. grammar            - Arabic morphological / syntactic errors
  3. mt_artifact        - translationese carried over from Google/IBM MT
  4. repetition         - degenerate / repeated or templated captions

We also compute a local, dependency-free ``repetition_ratio`` for every caption
so degenerate output is flagged statistically rather than only subjectively.

Usage:
    python3 llm_error_analysis.py --model qwen3:8b --limit 10 [--data-dir DIR]
"""
import argparse
import json
import os
import sys
import collections
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_helper import generate_json, normalize_host, slug_model   # noqa: E402
from data_loader import sample_dataset, load_xlsx_rows     # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT_JSON = os.path.join(RESULTS_DIR, "llm_error_analysis.json")
OUT_TXT = os.path.join(RESULTS_DIR, "llm_error_analysis_report.txt")


ERROR_PROMPT = (
    "You are an Arabic linguistics expert analysing machine-generated image "
    "captions for a research paper. Given an image description, reference "
    "captions, and one generated caption, identify concrete problems.\n\n"
    "Image description: {image}\n"
    "Reference captions:\n{references}\n"
    "Generated caption: {hypot}\n\n"
    "Categorise every problem found. Possible categories:\n"
    "  hallucination: object/attribute/person/action in the caption that is NOT "
    "            supported by the references or the image description\n"
    "  grammar: Arabic morphological or syntactic error (e.g. wrong gender/"
    "            number agreement, broken construct phrase)\n"
    "  mt_artifact: clearly machine-translated word order or a literal, unnatural "
    "            calque of English (translationese)\n"
    "  repetition: duplicated words or a generic templated phrase\n"
    "  other: anything else\n\n"
    "Return JSON ONLY: "
    "{{\"issues\": [{{\"category\": <one of the above>, \"span\": <quoted word(s) or "
    "\"caption\">, \"explanation\": <short Arabic/English reason>}}], "
    "\"overall_quality\": <0-5 float>, \"readability\": <0-5 float>}}"
)


def repetition_ratio(text):
    """Ratio of tokens that are exact repeats of a previous token (0..1)."""
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    seen = collections.Counter()
    repeats = 0
    for t in toks:
        if seen[t]:
            repeats += 1
        seen[t] += 1
    return repeats / len(toks)


def build_inputs(limit, xlsx=False, xlsx_path=None):
    inputs = []
    if xlsx:
        for rec in load_xlsx_rows(xlsx_path):
            base = {"image": f"an image (id {rec['key']}) from the MS COCO dataset",
                    "references": rec.get("eng_ref", [])}
            for ar in rec.get("ar_google", []):
                inputs.append({**base, "hypothesis": ar, "config": "ar_google"})
            for ar in rec.get("ar_manual", []):
                inputs.append({**base, "hypothesis": ar, "config": "ar_manual"})
        if limit > 0:
            inputs = inputs[:limit]
        return inputs
    for rec in sample_dataset().values():
        base = {
            "image": rec.get("img", "an image (see reference captions)"),
            "references": rec.get("ar_google_ref", []) + rec.get("eng_ref", []),
        }
        if rec.get("gen_ar_google"):
            inputs.append({**base, "hypothesis": rec["gen_ar_google"], "config": "ar_google"})
        if rec.get("gen_ar_ibm"):
            inputs.append({**base, "hypothesis": rec["gen_ar_ibm"], "config": "ar_ibm"})
        if rec.get("gen_en_ar"):
            inputs.append({**base, "hypothesis": rec["gen_en_ar"], "config": "en_ar"})
    if limit > 0:
        inputs = inputs[:limit]
    return inputs


def analyse_one(rec, model, host, verbose=False):
    rr = repetition_ratio(rec["hypothesis"])
    refs_text = "\n".join(f"- {r}" for r in rec["references"][:5]) or "- (none)"
    prompt = ERROR_PROMPT.format(image=rec["image"], references=refs_text,
                                 hypot=rec["hypothesis"])
    try:
        parsed = generate_json(prompt, model=model, host=host, num_predict=1024)
    except Exception as e:
        # A single bad response must never bring down the whole run: keep the
        # item but mark it as failed so it can be retried separately.
        return {"config": rec["config"], "hypothesis": rec["hypothesis"],
                "repetition_ratio": round(rr, 3), "issues": [], "error": str(e),
                "overall_quality": None, "readability": None}
    # The model sometimes returns a bare JSON array of issues (or otherwise
    # malformed JSON) instead of the full object -- never crash the run on that.
    if isinstance(parsed, dict):
        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        overall_quality = parsed.get("overall_quality")
        readability = parsed.get("readability")
    elif isinstance(parsed, list):
        issues = parsed
        overall_quality = readability = None
    else:
        issues = []
        overall_quality = readability = None
    for it in issues:
        if not isinstance(it, dict):
            issues = []
            break
    result = {
        "config": rec["config"],
        "hypothesis": rec["hypothesis"],
        "repetition_ratio": round(rr, 3),
        "issues": issues,
        "overall_quality": overall_quality,
        "readability": readability,
    }
    if verbose:
        print(f"  [{rec['config']}] RR={rr:.2f} issues={len(issues)} "
              f"q={result['overall_quality']} r={result['readability']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--xlsx", action="store_true",
                    help="read the full dataset from the paper's Excel workbook")
    ap.add_argument("--xlsx-path", default=None,
                    help="explicit path to the Excel workbook (defaults to data/arabic_captions.xlsx)")
    args = ap.parse_args()

    host = normalize_host(args.host)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    inputs = build_inputs(args.limit, args.xlsx, args.xlsx_path)
    print(f"Analysing {len(inputs)} caption(s) with '{args.model}'...")

    per_item, by_config = [], collections.defaultdict(list)
    for i, rec in enumerate(inputs, 1):
        print(f"[{i}/{len(inputs)}] config='{rec['config']}'")
        res = analyse_one(rec, args.model, host, args.verbose)
        per_item.append(res)
        by_config[rec["config"]].append(res)

    summary = {}
    for cfg, items in by_config.items():
        cats = collections.Counter()
        examples = collections.defaultdict(str)
        for it in items:
            for issue in it["issues"]:
                c = issue.get("category", "other")
                cats[c] += 1
                if not examples[c]:
                    examples[c] = f"  e.g. \"{issue.get('span','')}\" ({issue.get('explanation','')})"
        summary[cfg] = {
            "n": len(items),
            "mean_repetition_ratio": round(statistics.mean([x["repetition_ratio"]
                                                           for x in items]), 3),
            "mean_overall_quality": round(statistics.mean(
                [x["overall_quality"] or 0 for x in items]), 3),
            "issues_by_category": dict(cats),
            "examples": dict(examples),
        }

    tag = slug_model(args.model)
    out_json = os.path.join(RESULTS_DIR, f"llm_error_analysis__{tag}.json")
    out_txt = os.path.join(RESULTS_DIR, f"llm_error_analysis_report__{tag}.txt")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "per_item": per_item, "summary": summary},
                  f, ensure_ascii=False, indent=2)

    lines = [f"LLM Error Analysis (model: {args.model})", "=" * 60]
    for cfg, s in summary.items():
        lines.append(f"\n[{cfg}] (n={s['n']})")
        lines.append(f"  mean repetition ratio : {s['mean_repetition_ratio']}")
        lines.append(f"  mean overall quality  : {s['mean_overall_quality']}")
        lines.append(f"  issues by category    : "
                     f"{dict(s['issues_by_category']) or 'none detected'}")
        for c, ex in s["examples"].items():
            lines.append(f"    {ex}")
    report = "\n".join(lines)
    print("\n" + report)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    # Backward-compatible aliases (single default-model run behaviour).
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "per_item": per_item, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved JSON -> {out_json}\nSaved report -> {out_txt}")


if __name__ == "__main__":
    main()