#!/usr/bin/env python3
"""Experiment V3 - image-grounded LLM-as-Judge / error analysis (OLlama vision).

The text-only judge (Sec. LLM-Based Evaluation) scores the Google-MT Arabic
captions against the English references but never sees the image. Here the
vision LLM is given the actual image and the Arabic caption and scores it on
fluency / relevance / object accuracy / overall, flagging hallucinations that
are not supported by the *image* itself (not just the references).

Usage:
    python3 vlm_judge.py --model qwen3.8:27b --image-dir DIR [--limit N]
"""
import argparse
import json
import os
import sys
import statistics

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from ollama_helper import generate_vision, normalize_host, slug_model  # noqa: E402
from data_loader import load_xlsx_rows  # noqa: E402

DEFAULT_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(DEFAULT_REPO, "analysis", "results")
DEFAULT_XLSX = os.path.join(DEFAULT_REPO, "data", "arabic_captions.xlsx")
DEFAULT_IMAGES = os.path.join(DEFAULT_REPO, "data", "coco", "train2014")

JUDGE_PROMPT = (
    "You are an expert evaluator of Arabic image captions. You are given an "
    "image and one Arabic caption. Judge the caption against what is actually "
    "visible in the image.\n\n"
    "Arabic caption to evaluate: {caption}\n\n"
    "Score from 0 to 5 (0 = very poor, 5 = excellent):\n"
    "  fluency:          grammatical correctness and naturalness of the Arabic\n"
    "  relevance:        how well the caption describes the image content\n"
    "  object_accuracy:  whether every mentioned object/action is really in the "
    "image (penalise hallucinations strongly)\n"
    "  overall:          your holistic quality score\n\n"
    'Return JSON ONLY with keys "fluency", "relevance", "object_accuracy", '
    '"overall" (floats 0-5) and "hallucinations" (array of the hallucinated '
    'Arabic spans, empty array if none).'
)


def load_captions(limit, image_dir, xlsx_path):
    rows = []
    seen = set()
    dataset = load_xlsx_rows(xlsx_path)
    for rec in dataset:
        k = rec["key"]
        if k in seen:
            continue
        seen.add(k)
        img = os.path.join(image_dir, k)
        if not os.path.exists(img):
            continue
        for cap in rec.get("ar_google", []):
            rows.append({"key": k, "image": img, "caption": cap})
        if limit and len(rows) >= limit:
            break
    if limit and len(rows) > limit:
        rows = rows[:limit]
    return rows


def parse_scores(text):
    """Best-effort score extraction from the model JSON reply (returns dict)."""
    import json as _json
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "```").strip("`")
    try:
        data = _json.loads(text)
    except (_json.JSONDecodeError, ValueError):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            data = _json.loads(text[start:end + 1])
        except (_json.JSONDecodeError, ValueError):
            return None
    if not isinstance(data, dict):
        return None
    return {k: data.get(k) for k in
            ("fluency", "relevance", "object_accuracy", "overall",
             "hallucinations")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all captions")
    ap.add_argument("--image-dir", default=DEFAULT_IMAGES)
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    host = normalize_host(args.host)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = load_captions(args.limit, args.image_dir, args.xlsx)
    print(f"Judging {len(rows)} Arabic caption(s) against images with "
          f"'{args.model}'...")

    tag = slug_model(args.model)
    out_json = os.path.join(RESULTS_DIR, f"vlm_judge__{tag}.json")
    out_txt = os.path.join(RESULTS_DIR, f"vlm_judge__{tag}.txt")

    per_item, failures = [], 0
    for i, rec in enumerate(rows, 1):
        prompt = JUDGE_PROMPT.format(caption=rec["caption"])
        try:
            raw = generate_vision(prompt, model=args.model, images=[rec["image"]],
                                  host=host, num_predict=400)
            parsed = parse_scores(raw)
            if not parsed:
                raise ValueError(f"unparseable reply: {raw[:200]!r}")
            rec.update(parsed)
        except Exception as e:
            failures += 1
            rec.update({"fluency": None, "relevance": None,
                        "object_accuracy": None, "overall": None,
                        "hallucinations": None, "error": str(e)})
        per_item.append(rec)
        if args.verbose:
            print(f"[{i}/{len(rows)}] {rec['key']} "
                  f"flu={rec.get('fluency')} rel={rec.get('relevance')} "
                  f"obj={rec.get('object_accuracy')} overall={rec.get('overall')} "
                  f"hall={len(rec.get('hallucinations') or [])}")
        if i % 100 == 0:
            print(f"  [{i}/{len(rows)}] done (failures={failures})", flush=True)

    scored = [r for r in per_item
              if r.get("overall") not in (None, "") and not r.get("error")]
    summary = {}
    if scored:
        def _mean(dim):
            vals = [float(r[dim]) for r in scored if r.get(dim) not in (None, "")]
            return round(sum(vals) / len(vals), 3) if vals else None
        summary = {"n": len(scored),
                   "fluency": _mean("fluency"),
                   "relevance": _mean("relevance"),
                   "object_accuracy": _mean("object_accuracy"),
                   "overall": _mean("overall"),
                   "mean_hallucinations_per_caption": round(
                       statistics.mean(len(r.get("hallucinations") or [])
                                       for r in scored), 3)}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "n": len(per_item),
                   "failures": failures, "summary": summary,
                   "per_item": per_item}, f, ensure_ascii=False, indent=2)

    lines = [f"Image-grounded LLM judge (model: {args.model})", "=" * 60,
             f"captions judged: {len(per_item)} (scored: {len(scored)}, "
             f"failures: {failures})"]
    if summary:
        lines.append(
            f"overall={summary['overall']}  fluency={summary['fluency']}  "
            f"relevance={summary['relevance']}  "
            f"object_accuracy={summary['object_accuracy']}  "
            f"mean hallucinations/caption={summary['mean_hallucinations_per_caption']}")
    report = "\n".join(lines)
    print("\n" + report)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved JSON -> {out_json}\nSaved report -> {out_txt}")


if __name__ == "__main__":
    main()