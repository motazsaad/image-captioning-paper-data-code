#!/usr/bin/env python3
"""Measure consistency between a VLM's English and Arabic captions.

For each shared COCO image, the Arabic VLM caption is back-translated to
English, then compared with the direct English VLM caption using embedding
cosine, token F1, BLEU-1, and an optional LLM semantic-consistency score.
The script checkpoints after each item, making it safe to resume after a GPU
workstation interruption.
"""
import argparse
import collections
import json
import math
import os
import re
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_helper import embed, cosine, generate, generate_json, normalize_host, slug_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "analysis", "results")

BACKTRANSLATE_PROMPT = """Back-translate this Arabic image caption into natural English.
Output only the English caption, with no explanation:
{arabic}"""

JUDGE_PROMPT = """You evaluate whether two English image captions describe the same
visual content. Ignore harmless wording, tense, article, and detail differences.
Penalise contradictory objects, actions, counts, locations, or relations.

Direct English VLM caption: {english}
Arabic VLM caption after back-translation: {backtranslated}

Return JSON only: {{"consistency": <number from 0 to 5>,
"rationale": "<brief explanation>"}}."""


def tokens(text):
    return [t.lower() for t in re.split(r"[^A-Za-z0-9']+", text or "") if t]


def token_f1(a, b):
    aa, bb = collections.Counter(tokens(a)), collections.Counter(tokens(b))
    overlap = sum((aa & bb).values())
    total = sum(aa.values()) + sum(bb.values())
    return 2 * overlap / total if total else 0.0


def bleu1(hypothesis, reference):
    hyp, ref = tokens(hypothesis), tokens(reference)
    if not hyp or not ref:
        return 0.0
    h, r = collections.Counter(hyp), collections.Counter(ref)
    precision = sum((h & r).values()) / len(hyp)
    brevity = math.exp(1 - len(ref) / len(hyp)) if len(hyp) < len(ref) else 1.0
    return precision * brevity


def read_items(path):
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


def summary(items):
    fields = ("embedding_cosine", "token_f1", "bleu1", "judge_consistency")
    out = {"n": len(items), "errors": sum(bool(x.get("error")) for x in items)}
    for field in fields:
        values = [x[field] for x in items if isinstance(x.get(field), (int, float))]
        out[f"mean_{field}"] = round(statistics.mean(values), 4) if values else None
        out[f"median_{field}"] = round(statistics.median(values), 4) if values else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--english-json", required=True, help="VLM English caption JSON")
    ap.add_argument("--arabic-json", required=True, help="VLM Arabic caption JSON")
    ap.add_argument("--translation-model", default="ministral-3:14b")
    ap.add_argument("--embed-model", default="qwen3-embedding:4b")
    ap.add_argument("--judge-model", default="ministral-3:14b",
                    help="set to empty string with --skip-judge to omit semantic judging")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--restart", action="store_true", help="discard an existing checkpoint")
    args = ap.parse_args()

    en_data, en_by_key = read_items(args.english_json)
    ar_data, ar_by_key = read_items(args.arabic_json)
    shared = [k for k in en_by_key if k in ar_by_key]
    if args.limit:
        shared = shared[:args.limit]
    if not shared:
        raise SystemExit("No shared image keys with captions found.")
    tag = slug_model(en_data.get("model", "vlm"))
    out_path = args.out or os.path.join(RESULTS_DIR, f"crosslingual_consistency__{tag}.json")
    previous = {}
    if os.path.exists(out_path) and not args.restart:
        with open(out_path, encoding="utf-8") as f:
            previous = {str(x.get("key")): x for x in json.load(f).get("per_item", [])}
    host = normalize_host(args.host)

    records = [previous[k] for k in shared if k in previous]
    for index, key in enumerate(shared, 1):
        if key in previous:
            continue
        en, ar = en_by_key[key], ar_by_key[key]
        rec = {"key": key, "english_caption": en.get("caption"),
               "arabic_caption": ar.get("caption"), "eng_refs": en.get("eng_refs", []),
               "ar_google": ar.get("ar_google", []), "backtranslated_en": None,
               "embedding_cosine": None, "token_f1": None, "bleu1": None,
               "judge_consistency": None, "judge_rationale": None, "error": None}
        try:
            back = generate(BACKTRANSLATE_PROMPT.format(arabic=rec["arabic_caption"]),
                            model=args.translation_model, host=host, num_predict=120)
            rec["backtranslated_en"] = back
            rec["embedding_cosine"] = round(cosine(
                embed(rec["english_caption"], model=args.embed_model, host=host),
                embed(back, model=args.embed_model, host=host)) or 0.0, 4)
            rec["token_f1"] = round(token_f1(rec["english_caption"], back), 4)
            rec["bleu1"] = round(bleu1(back, rec["english_caption"]), 4)
            if not args.skip_judge:
                judged = generate_json(JUDGE_PROMPT.format(
                    english=rec["english_caption"], backtranslated=back),
                    model=args.judge_model, host=host, num_predict=180)
                rec["judge_consistency"] = float(judged.get("consistency", 0) or 0)
                rec["judge_rationale"] = str(judged.get("rationale", ""))
        except Exception as exc:
            rec["error"] = str(exc)
        records.append(rec)
        payload = {"experiment": "cross-lingual VLM caption consistency", "english_model": en_data.get("model"),
                   "arabic_model": ar_data.get("model"), "translation_model": args.translation_model,
                   "embed_model": args.embed_model, "judge_model": None if args.skip_judge else args.judge_model,
                   "n_requested": len(shared), "per_item": records, "summary": summary(records)}
        atomic_write(out_path, payload)
        print(f"[{index}/{len(shared)}] {key}", flush=True)
    print(json.dumps(summary(records), indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
