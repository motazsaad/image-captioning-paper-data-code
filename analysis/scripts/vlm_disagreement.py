#!/usr/bin/env python3
"""Rank images by disagreement between two VLM caption outputs.

Embeddings from a local Ollama model quantify semantic agreement.  High
embedding distance identifies images where two independent zero-shot VLMs
describe materially different content, providing an annotation-free uncertainty
set for analysis.  Output is checkpointed after each image and resumable.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_helper import embed, cosine, normalize_host, slug_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "analysis", "results")


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, {str(x["key"]): x for x in data.get("per_item", []) if x.get("key")}


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


def aggregate(records):
    values = [x["disagreement"] for x in records if isinstance(x.get("disagreement"), (float, int))]
    return {"n": len(records), "errors": sum(bool(x.get("error")) for x in records),
            "mean_disagreement": round(statistics.mean(values), 4) if values else None,
            "median_disagreement": round(statistics.median(values), 4) if values else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captions-a", required=True, help="first VLM caption JSON")
    ap.add_argument("--captions-b", required=True, help="second VLM caption JSON")
    ap.add_argument("--embed-model", default="qwen3-embedding:4b")
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    a_data, a = load(args.captions_a)
    b_data, b = load(args.captions_b)
    keys = [k for k in a if k in b]
    if args.limit:
        keys = keys[:args.limit]
    if not keys:
        raise SystemExit("No shared image keys with captions found.")
    tag = f"{slug_model(a_data.get('model', 'a'))}__vs__{slug_model(b_data.get('model', 'b'))}__{a_data.get('lang', 'x')}"
    out_path = args.out or os.path.join(RESULTS_DIR, f"vlm_disagreement__{tag}.json")
    old = {}
    if os.path.exists(out_path) and not args.restart:
        with open(out_path, encoding="utf-8") as f:
            old = {str(x.get("key")): x for x in json.load(f).get("per_item", [])}
    host = normalize_host(args.host)
    records = [old[k] for k in keys if k in old]
    for i, key in enumerate(keys, 1):
        if key in old:
            continue
        ca, cb = a[key].get("caption"), b[key].get("caption")
        rec = {"key": key, "caption_a": ca, "caption_b": cb,
               "embedding_cosine": None, "disagreement": None,
               "length_delta_chars": abs(len(ca or "") - len(cb or "")), "error": None}
        try:
            sim = cosine(embed(ca, model=args.embed_model, host=host),
                         embed(cb, model=args.embed_model, host=host))
            rec["embedding_cosine"] = round(sim, 4) if sim is not None else None
            rec["disagreement"] = round(1 - sim, 4) if sim is not None else None
        except Exception as exc:
            rec["error"] = str(exc)
        records.append(rec)
        payload = {"experiment": "VLM disagreement as uncertainty", "model_a": a_data.get("model"),
                   "model_b": b_data.get("model"), "language": a_data.get("lang"),
                   "embed_model": args.embed_model, "per_item": records,
                   "summary": aggregate(records)}
        atomic_write(out_path, payload)
        print(f"[{i}/{len(keys)}] {key}", flush=True)
    ranked = sorted((x for x in records if x.get("disagreement") is not None),
                    key=lambda x: x["disagreement"], reverse=True)
    payload = {"experiment": "VLM disagreement as uncertainty", "model_a": a_data.get("model"),
               "model_b": b_data.get("model"), "language": a_data.get("lang"),
               "embed_model": args.embed_model, "per_item": records,
               "summary": aggregate(records), "top_disagreements": ranked[:args.top_k]}
    atomic_write(out_path, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
