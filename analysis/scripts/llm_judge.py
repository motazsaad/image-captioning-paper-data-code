#!/usr/bin/env python3
"""Suggestion 1 - LLM-as-Judge Semantic Evaluation.

The paper evaluates generated captions only with n-gram metrics (BLEU/ROUGE/
METEOR/CIDEr) which correlate weakly with human judgement, especially for
morphologically-rich Arabic. This script uses a local LLM (via Ollama) as a
reference-aware judge to score each already-generated caption on several
dimensions, producing a semantic evaluation that complements (and can be
correlated with) the paper's automatic metrics.

Dimensions (0-5 score each):
  1) fluency       - how grammatically natural / fluent the Arabic is
  2) relevance     - how well the caption matches the reference description
  3) object_accuracy- whether the objects mentioned are supported by references
                    (i.e. no hallucination)
  4) overall       - overall caption quality

We evaluate several system configurations, mirroring the paper's experiment
family:
  - ar_google (Exp2 hypothesis, generated Arabic model, Google-MT training)
  - ar_ibm    (Exp2 hypothesis, generated Arabic model, IBM-MT training)
  - en_ar     (Exp3/5 hypothesis, English output translated to Arabic)

The official English model hypothesis (Exp1) is scored too for the English
baseline.

Usage:
    python3 llm_judge.py --model qwen3:8b --limit 5 [--data-dir DIR]
Writes per-config mean scores to llm_improvements/results/llm_judge_results.json
and a human-readable report.
"""
import argparse
import json
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_helper import generate_json, normalize_host, slug_model  # noqa: E402
from data_loader import (load_caption_file,            # noqa: E402
                         sample_dataset, describe_available,
                         load_xlsx_rows)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT_JSON = os.path.join(RESULTS_DIR, "llm_judge_results.json")
OUT_TXT = os.path.join(RESULTS_DIR, "llm_judge_report.txt")


JUDGE_PROMPT = (
    "You are an expert evaluator of Arabic image caption quality for a research "
    "paper. You are given an image, its human reference captions, and one "
    "machine-generated caption. Judge only the generated caption.\n\n"
    "Image description: {image}\n"
    "Reference captions:\n{references}\n"
    "Generated caption (to evaluate): {hypot}\n\n"
    "Score the generated caption from 0 to 5 (0 = very poor, 5 = excellent) on "
    "each dimension:\n"
    "  fluency:          grammatical correctness and naturalness of the Arabic\n"
    "  relevance:        how well it matches the content of the references\n"
    "  object_accuracy:  whether objects are grounded in the references "
    "(penalise hallucinations unsupported by the references)\n"
    "  overall:          your holistic quality score\n\n"
    "Return a JSON object ONLY, with keys 'fluency', 'relevance', "
    "'object_accuracy', 'overall'. Each value must be a float between 0 and 5."
)


def build_inputs(limit, data_dir, xlsx=False, xlsx_path=None):
    """Return a list of records: image, references, hypothesis, config tag."""
    inputs = []

    # Optional full-dataset path: evaluate the paper's workbook rows directly.
    if xlsx:
        for rec in load_xlsx_rows(xlsx_path):
            refs = rec.get("eng_ref", [])
            for ar in rec.get("ar_google", []):
                inputs.append({"image": f"an image (id {rec['key']}) from the MS COCO dataset",
                               "references": refs,
                               "hypothesis": ar,
                               "config": "ar_google",
                               "key": rec["key"]})
            for ar in rec.get("ar_manual", []):
                inputs.append({"image": f"an image (id {rec['key']}) from the MS COCO dataset",
                               "references": refs,
                               "hypothesis": ar,
                               "config": "ar_manual",
                               "key": rec["key"]})
        if limit > 0:
            inputs = inputs[:limit]
        return inputs

    # Try to build real samples from the pickle artifacts when present.
    gen_en = load_caption_file("image_path_to_caption_gen_en.sav", [data_dir] if data_dir else None)
    gen_en_ar = load_caption_file("image_path_to_caption_gen_en_ar.sav", [data_dir] if data_dir else None)
    gen_ar = load_caption_file("image_path_to_caption_gen_ar.sav", [data_dir] if data_dir else None)
    ar_google = load_caption_file("image_path_to_caption_ar.sav", [data_dir] if data_dir else None)
    ar_ibm = load_caption_file("image_path_to_caption_ar_ibm.sav", [data_dir] if data_dir else None)
    eng_ref = load_caption_file("image_path_to_caption_en.sav", [data_dir] if data_dir else None)

    available = describe_available([data_dir] if data_dir else None)
    if available:
        # Real-data path: generate a few random keys that exist across files.
        keys = list((gen_ar or gen_en_ar or gen_en or {}).keys())
        keys = keys[:limit]
        image_desc = "an image from the MS COCO dataset (see reference captions)"
        for k in keys:
            refs = (ar_google or {}).get(k) or (eng_ref or {}).get(k) or []
            if gen_ar and k in gen_ar:
                inputs.append({"image": image_desc, "references": refs,
                               "hypothesis": gen_ar[k][0], "config": "ar_google"})
            if gen_en_ar and k in gen_en_ar:
                inputs.append({"image": image_desc, "references": refs,
                               "hypothesis": gen_en_ar[k][0], "config": "en_ar"})
            if gen_en and k in gen_en:
                inputs.append({"image": image_desc, "references": refs,
                               "hypothesis": gen_en[k][0], "config": "en_en"})
    else:
        # Fallback: built-in sample (paper Table 3).
        for rec in sample_dataset().values():
            image_desc = rec.get("img", "an image (see reference captions)")
            refs = rec.get("ar_google_ref", []) + rec.get("eng_ref", [])
            base = {"image": image_desc, "references": refs}
            if rec.get("gen_ar_google"):
                inputs.append({**base, "hypothesis": rec["gen_ar_google"], "config": "ar_google"})
            if rec.get("gen_ar_ibm"):
                inputs.append({**base, "hypothesis": rec["gen_ar_ibm"], "config": "ar_ibm"})
            if rec.get("gen_en_ar"):
                inputs.append({**base, "hypothesis": rec["gen_en_ar"], "config": "en_ar"})

    if not inputs:
        raise SystemExit("No inputs available. Provide --data-dir with the .sav artifacts.")
    if limit and limit > 0:
        inputs = inputs[:limit]
    return inputs


def judge_one(rec, model, host, verbose=False):
    refs_text = "\n".join(f"- {r}" for r in rec["references"][:5]) or "- (none)"
    prompt = JUDGE_PROMPT.format(
        image=rec["image"], references=refs_text, hypot=rec["hypothesis"])
    try:
        parsed = generate_json(prompt, model=model, host=host, num_predict=300)
    except Exception as e:
        # A single bad/malformed response must never kill a full run.
        return {"config": rec["config"], "hypothesis": rec["hypothesis"],
                "fluency": 0, "relevance": 0, "object_accuracy": 0,
                "overall": 0, "error": str(e)}
    if not isinstance(parsed, dict):
        parsed = {}
    scores = {
        "fluency": float(parsed.get("fluency", 0) or 0),
        "relevance": float(parsed.get("relevance", 0) or 0),
        "object_accuracy": float(parsed.get("object_accuracy", 0) or 0),
        "overall": float(parsed.get("overall", 0) or 0),
    }
    if verbose:
        print(f"  [{rec['config']}] {rec['hypothesis']!r}\n"
              f"      -> {scores}")
    return scores


def main():
    ap = argparse.ArgumentParser(description="LLM-as-Judge evaluation of generated captions")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all available")
    ap.add_argument("--data-dir", default=None,
                    help="directory containing the .sav caption pickles")
    ap.add_argument("--xlsx", action="store_true",
                    help="read the full dataset from the paper's Excel workbook")
    ap.add_argument("--xlsx-path", default=None,
                    help="explicit path to the Excel workbook (defaults to data/arabic_captions.xlsx)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    host = normalize_host(args.host)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    inputs = build_inputs(args.limit, args.data_dir, args.xlsx, args.xlsx_path)
    if args.limit > 0:
        inputs = inputs[:args.limit]
    print(f"Evaluating {len(inputs)} caption(s) with model '{args.model}'...")

    agg = {}          # config -> {dim: [scores]}
    per_item = []
    for i, rec in enumerate(inputs, 1):
        print(f"[{i}/{len(inputs)}] judging config='{rec['config']}'")
        scores = judge_one(rec, args.model, host, args.verbose)
        agg.setdefault(rec["config"], {}).setdefault("fluency", []).append(scores["fluency"])
        agg.setdefault(rec["config"], {}).setdefault("relevance", []).append(scores["relevance"])
        agg.setdefault(rec["config"], {}).setdefault("object_accuracy", []).append(scores["object_accuracy"])
        agg.setdefault(rec["config"], {}).setdefault("overall", []).append(scores["overall"])
        per_item.append({"config": rec["config"], "hypothesis": rec["hypothesis"], **scores})

    summary = {}
    for config, dims in agg.items():
        summary[config] = {d: statistics.mean(v) for d, v in dims.items()}

    tag = slug_model(args.model)
    out_json = os.path.join(RESULTS_DIR, f"llm_judge_results__{tag}.json")
    out_txt = os.path.join(RESULTS_DIR, f"llm_judge_report__{tag}.txt")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "per_item": per_item, "summary": summary},
                  f, ensure_ascii=False, indent=2)

    lines = [f"LLM-as-Judge evaluation (model: {args.model})",
             "=" * 60,
             f"{'config':<12}{'n':<5}{'fluency':<9}{'relevance':<10}"
             f"{'obj_acc':<10}{'overall':<8}"]
    for config, dims in summary.items():
        n = len(agg[config]["overall"])
        lines.append(f"{config:<12}{n:<5}"
                     f"{dims['fluency']:<9.2f}{dims['relevance']:<10.2f}"
                     f"{dims['object_accuracy']:<10.2f}{dims['overall']:<8.2f}")
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
    print(f"\nSaved JSON -> {out_json}")
    print(f"Saved report -> {out_txt}")


if __name__ == "__main__":
    main()
