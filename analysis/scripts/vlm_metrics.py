#!/usr/bin/env python3
"""Compute COCO captioning metrics for VLM-generated captions.

Runs with the venv that has pycocoevalcap:
    ~/venvs/capcaption/bin/python vlm_metrics.py ...

Reads a vlm_caption results JSON and scores the generated captions against the
chosen reference set (eng_refs for V1 English captions; ar_google or ar_manual
for V2 Arabic captions). Reports BLEU-1..4, METEOR, CIDEr, ROUGE-L and saves a
JSON with the same official COCO-evalcap computation used by the paper.

Note: METEOR shells out to a Java jar and is slow on large sets.
"""
import argparse
import json
import os
import sys
import time

DEFAULT_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(DEFAULT_REPO, "analysis", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption-json", required=True,
                    help="vlm_caption results JSON (per_item with captions)")
    ap.add_argument("--refs", choices=["eng_refs", "ar_google", "ar_manual"],
                    default="eng_refs")
    ap.add_argument("--no-meteor", action="store_true",
                    help="skip METEOR (Java jar) for a faster run")
    args = ap.parse_args()

    with open(args.caption_json, encoding="utf-8") as f:
        data = json.load(f)

    gts, res = {}, {}
    skipped = 0
    for i, it in enumerate(data["per_item"]):
        cap = (it.get("caption") or "").strip()
        refs = [r for r in (it.get(args.refs) or []) if r]
        if not cap or not refs:
            skipped += 1
            continue
        gts[i] = refs
        res[i] = [cap]
    print(f"Scoring {len(res)} captions (skipped {skipped}) against '{args.refs}'")

    out = {"model": data.get("model"), "lang": data.get("lang"),
           "refs": args.refs, "n": len(res), "metrics": {}, "per_image": {}}

    if not res:
        # No captions have references for this set (e.g. empty manual refs);
        # pycocoevalcap crashes on empty input, so record an empty result instead.
        print("  WARNING: no captions with references; writing empty metrics")
        out["metrics"] = {"BLEU-1": None, "BLEU-2": None, "BLEU-3": None,
                          "BLEU-4": None, "BLEUavg": None, "ROUGE-L": None,
                          "CIDEr": None, "METEOR": None}
        out_json = args.caption_json.replace(".json", f"__metrics__{args.refs}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved -> {out_json}")
        return

    bleu = Bleu(4)
    score, scores = bleu.compute_score(gts, res)
    for n in range(1, 5):
        out["metrics"][f"BLEU-{n}"] = round(float(score[n - 1]), 4)
    out["metrics"]["BLEUavg"] = round(sum(float(s) for s in score) / 4, 4)
    bleu1_per_image = scores[0]

    rouge = Rouge()
    score, scores = rouge.compute_score(gts, res)
    out["metrics"]["ROUGE-L"] = round(float(score), 4)
    rouge_per_image = scores

    cider = Cider()
    score, scores = cider.compute_score(gts, res)
    out["metrics"]["CIDEr"] = round(float(score), 4)
    cider_per_image = scores

    if not args.no_meteor:
        t0 = time.time()
        meteor = Meteor()
        score, scores = meteor.compute_score(gts, res)
        out["metrics"]["METEOR"] = round(float(score), 4)
        print(f"  (METEOR took {time.time() - t0:.0f}s)")
        meteor_per_image = scores
    else:
        meteor_per_image = {}

    for i in res:
        out["per_image"][str(i)] = {
            "bleu1": round(float(bleu1_per_image[i]), 4),
            "rouge": round(float(rouge_per_image[i]), 4),
            "cider": round(float(cider_per_image[i]), 4),
            "meteor": round(float(meteor_per_image[i]), 4)
            if meteor_per_image else None,
        }

    out_json = args.caption_json.replace(".json", f"__metrics__{args.refs}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"model={out['model']} lang={out['lang']} refs={args.refs} n={out['n']}")
    for k, v in out["metrics"].items():
        print(f"  {k:<10} {v}")
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    try:
        from pycocoevalcap.meteor.meteor import Meteor
    except Exception as e:  # pragma: no cover
        print(f"warning: Meteor unavailable ({e})")
        Meteor = None
    main()