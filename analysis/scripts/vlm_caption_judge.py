#!/usr/bin/env python3
"""Image-grounded judge for generated VLM caption JSON files.

Unlike vlm_judge.py, which evaluates the Google-MT reference corpus, this
script evaluates captions produced by a VLM prompt condition. It checkpoints
after each image and is designed for standard/concise/grounded prompt ablations.
"""
import argparse
import json
import os
import sys
import tempfile
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_helper import generate_vision, normalize_host, slug_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "analysis", "results")
DEFAULT_IMAGES = os.path.join(REPO_ROOT, "data", "coco", "train2014")
PROMPT = """You are an expert evaluator of Arabic image captions. You receive an image
and one Arabic caption. Judge the caption against what is actually visible.

Arabic caption: {caption}

Score 0 to 5 (0=very poor, 5=excellent):
- fluency: grammatical correctness and naturalness of Arabic;
- relevance: how well the caption describes the image;
- object_accuracy: whether mentioned objects/actions are visible; penalise
  hallucinations strongly;
- overall: holistic quality.

Return JSON only with keys fluency, relevance, object_accuracy, overall, and
hallucinations (an array of unsupported Arabic spans, or an empty array)."""


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


def parse(raw):
    try:
        data = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except (json.JSONDecodeError, ValueError):
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return {"fluency": data.get("fluency"), "relevance": data.get("relevance"),
            "object_accuracy": data.get("object_accuracy"), "overall": data.get("overall"),
            "hallucinations": data.get("hallucinations") or []}


def summarise(records):
    scored = [r for r in records if not r.get("error") and r.get("overall") is not None]
    result = {"n": len(scored), "failures": len(records) - len(scored)}
    for field in ("fluency", "relevance", "object_accuracy", "overall"):
        values = [float(x[field]) for x in scored if x.get(field) is not None]
        result[field] = round(statistics.mean(values), 3) if values else None
    result["mean_hallucinations_per_caption"] = round(statistics.mean(
        len(x.get("hallucinations") or []) for x in scored), 3) if scored else None
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caption-json", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--image-dir", default=DEFAULT_IMAGES)
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    with open(args.caption_json, encoding="utf-8") as f:
        source = json.load(f)
    items = [x for x in source.get("per_item", []) if x.get("key") and x.get("caption")]
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise SystemExit("No captions found.")
    source_tag = os.path.basename(args.caption_json).removesuffix(".json").replace("vlm_caption__", "")
    out_path = args.out or os.path.join(
        RESULTS_DIR, f"vlm_caption_judge__{slug_model(args.judge_model)}__{source_tag}.json")
    previous = {}
    if os.path.exists(out_path) and not args.restart:
        with open(out_path, encoding="utf-8") as f:
            previous = {str(x.get("key")): x for x in json.load(f).get("per_item", [])}
    host = normalize_host(args.host)
    records = [previous[str(x["key"])] for x in items if str(x["key"]) in previous]
    for index, item in enumerate(items, 1):
        key = str(item["key"])
        if key in previous:
            continue
        image = os.path.join(args.image_dir, key)
        rec = {"key": key, "caption": item["caption"], "fluency": None,
               "relevance": None, "object_accuracy": None, "overall": None,
               "hallucinations": [], "error": None}
        try:
            if not os.path.exists(image):
                raise FileNotFoundError(image)
            parsed = parse(generate_vision(PROMPT.format(caption=item["caption"]),
                                           model=args.judge_model, images=[image], host=host,
                                           num_predict=320))
            if not parsed:
                raise ValueError("unparseable judge response")
            rec.update(parsed)
        except Exception as exc:
            rec["error"] = str(exc)
        records.append(rec)
        atomic_write(out_path, {"experiment": "image-grounded VLM prompt evaluation",
                     "caption_model": source.get("model"), "caption_run_tag": source.get("run_tag"),
                     "caption_prompt": source.get("prompt"), "judge_model": args.judge_model,
                     "per_item": records, "summary": summarise(records)})
        print(f"[{index}/{len(items)}] {key}", flush=True)
    final = {"experiment": "image-grounded VLM prompt evaluation",
             "caption_model": source.get("model"), "caption_run_tag": source.get("run_tag"),
             "caption_prompt": source.get("prompt"), "judge_model": args.judge_model,
             "per_item": records, "summary": summarise(records)}
    atomic_write(out_path, final)
    print(json.dumps(final["summary"], indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
