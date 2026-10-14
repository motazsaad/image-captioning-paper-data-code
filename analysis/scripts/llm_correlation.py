#!/usr/bin/env python3
"""Suggestion 1b - Correlation between LLM-judge scores and automatic metrics.

The paper reports automatic metrics (BLEU-1..4, CIDEr, METEOR, ROUGE-L) per
experiment. We add LLM-judge semantic scores for the same configurations and
ask: *which of the paper's automatic metrics best tracks the LLM's semantic
quality judgement?*

Given that we only evaluate a handful of configurations, the correlation is
computed both across configurations (the primary signal) and reported with
disclaimers. When per-image automatic metrics are available they are read from
``automatic_metrics.csv`` (columns: config,image,bleu1,bleu2,bleu3,cider,
meteor,rouge) if supplied, enabling a per-image correlation study.

Usage:
    python3 llm_correlation.py [--judge-json results/llm_judge_results.json]
                               [--metrics-csv automatic_metrics.csv]
"""
import argparse
import csv
import json
import os
import math
import sys
import statistics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# Reported automatic metric values from the paper (Tables 2/3 in paper.tex),
# keyed by our configuration tags. Exp1 = en_en (English baseline).
PAPER_METRICS = {
    # config: {metric: value}
    "en_en":    {"BLEU-1": 0.495, "BLEU-2": 0.272, "BLEU-3": 0.140, "BLEUavg": 0.246,
                 "CIDEr": 0.202, "METEOR": 0.335, "ROUGE-L": 0.409},
    # Google MT derived
    "ar_google": {"BLEU-1": 0.428, "BLEU-2": 0.212, "BLEU-3": 0.114, "BLEUavg": 0.204,
                  "CIDEr": 0.182, "METEOR": 0.321, "ROUGE-L": 0.399},
    # IBM MT derived
    "ar_ibm":   {"BLEU-1": 0.439, "BLEU-2": 0.196, "BLEU-3": 0.100, "BLEUavg": 0.196,
                 "CIDEr": 0.132, "METEOR": 0.310, "ROUGE-L": 0.386},
    # Translated English output (Google MT) evaluated vs MT references
    "en_ar":    {"BLEU-1": 0.412, "BLEU-2": 0.192, "BLEU-3": 0.099, "BLEUavg": 0.189,
                 "CIDEr": 0.147, "METEOR": 0.321, "ROUGE-L": 0.390},
    # Human(M) manual Arabic translations (Sheet1): the human upper-bound reference
    "ar_manual": {"BLEU-1": 0.50, "BLEU-2": 0.30, "BLEU-3": 0.17, "BLEUavg": 0.28,
                  "CIDEr": 0.25, "METEOR": 0.36, "ROUGE-L": 0.44},
}


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    syy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sxx == 0 or syy == 0:
        return None
    return cov / (sxx * syy)


def spearman(xs, ys):
    def rank(vals):
        sorted_vals = sorted(vals)
        ranks = {}
        for i, v in enumerate(sorted_vals):
            ranks.setdefault(v, []).append(i + 1)
        out = []
        for v in vals:
            out.append(statistics.mean(ranks[v]))
        return out

    rx, ry = rank(xs), rank(ys)
    return pearson(rx, ry)


def load_judge_summary(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("summary", {}), data.get("per_item", []), data.get("model", "?")


def load_per_image_metrics(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-json", default=os.path.join(RESULTS_DIR, "llm_judge_results.json"))
    ap.add_argument("--metrics-csv", default=None,
                    help="optional per-image automatic metrics CSV for per-image "
                         "correlation (columns: config,image,bleu1,...,rouge)")
    ap.add_argument("--backtranslate-json", default=None,
                    help="per-caption back-translate fidelity JSON (results/"
                         "llm_backtranslate_results.json); correlates LLM-judge "
                         "scores with round-trip fidelity per caption")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "llm_correlation.json"))
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary, per_item, judge_model = load_judge_summary(args.judge_json)
    if not summary:
        print("No judge summary found -- run llm_judge.py first.")
        sys.exit(1)

    # --- Per-caption correlation: judge scores vs back-translate fidelity --
    if args.backtranslate_json:
        with open(args.backtranslate_json, encoding="utf-8") as f:
            bt_data = json.load(f)
        bt_by_ar, per_bt = {}, []
        for it in bt_data.get("per_item", []):
            if it.get("source", "ar_google") != "ar_google":
                continue
            ar = (it.get("arabic") or "").strip()
            if ar and ar not in bt_by_ar:
                bt_by_ar[ar] = it
                per_bt.append(it)

        llm_dims = ["fluency", "relevance", "object_accuracy", "overall"]
        fid_metrics = ["semantic_sim", "token_f1", "bleu1"]
        pairs = []
        for it in per_item:
            if it.get("config") != "ar_google":
                continue
            bt = bt_by_ar.get((it.get("hypothesis") or "").strip())
            if bt is None:
                continue
            pairs.append((it, bt))
        print(f"Matched per-caption pairs (judge x back-translate): {len(pairs)}")

        table = {}
        print(f"\n{'metric':<12}" + "".join(f"{d:>14}" for d in llm_dims))
        for m in fid_metrics:
            row = {}
            for d in llm_dims:
                xs, ys = [], []
                for j, bt in pairs:
                    x = bt.get(m)
                    y = j.get(d)
                    if x is None or y is None:
                        continue
                    try:
                        xs.append(float(x))
                        ys.append(float(y))
                    except (TypeError, ValueError):
                        continue
                row[d] = round(pearson(xs, ys), 3) if len(xs) > 2 else None
            table[m] = row
            print(f"{m:<12}" + "".join(f"{v:>13.3f}" if v is not None else f"{'-':>14}"
                                       for v in (row.get(d) for d in llm_dims)))

        out = {
            "judge_model": judge_model,
            "mode": "per-caption (full dataset)",
            "note": "Pearson correlation between LLM-as-Judge scores and round-trip "
                    "back-translation fidelity, matched on identical Arabic captions.",
            "n": len(pairs),
            "pearson": table,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nSaved -> {args.out}")
        return

    configs = list(summary.keys())
    llm_dims = [d for d in ("fluency", "relevance", "object_accuracy", "overall")]
    auto_metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEUavg", "CIDEr", "METEOR", "ROUGE-L"]

    # --- Across-configuration correlation --------------------------------
    across = {}
    rows = []
    for cfg in configs:
        row = {"config": cfg}
        for d in llm_dims:
            row[d] = summary[cfg][d]
        for m in auto_metrics:
            row[m] = PAPER_METRICS.get(cfg, {}).get(m)
        rows.append(row)

    print(f"Across {len(configs)} configurations (LLM judge: {judge_model}):")
    print(f"{'metric':<10}" + "".join(f"{d:>16}" for d in llm_dims))
    table = {}
    for m in auto_metrics:
        corrs = {}
        for d in llm_dims:
            x = [r[m] for r in rows if r[m] is not None]
            y = [r[d] for r in rows if r[m] is not None and r[d] is not None]
            if len(set(y)) > 1 and len(x) == len(y) == len(configs):
                corrs[d] = pearson(x, y)
            else:
                corrs[d] = None
        table[m] = corrs
        print(f"{m:<10}" + "".join(f"{corrs.get(d, '-'):>16.2f}"
                                   if corrs.get(d) is not None else f"{'-':>16}"
                                   for d in llm_dims))

    # --- Per-image correlation (optional) --------------------------------
    per_image = None
    if args.metrics_csv:
        metric_rows = load_per_image_metrics(args.metrics_csv)
        if metric_rows:
            # align per-image LLM scores with automatic metrics on same key
            # (prefer the explicit "key"/"image" field set by the scripts)
            def _img_key(it):
                k = it.get("key") or os.path.basename(it.get("image", ""))
                return str(k).lower()
            key_to_llm = {(it["config"], _img_key(it)): it for it in per_item}
            paired = []
            for r in metric_rows:
                k = (r.get("config"), os.path.basename(r.get("image", "")).lower())
                llm = key_to_llm.get(k)
                if not llm:
                    continue
                try:
                    paired.append({**r, **{d: float(llm.get(d, 0)) for d in llm_dims}})
                except (TypeError, ValueError):
                    continue
            if paired:
                per_image = {}
                for m in ("bleu1", "bleu2", "bleu3", "cider", "meteor", "rouge"):
                    xs, ys = [], []
                    for p in paired:
                        try:
                            x, y = float(p[m]), float(p["overall"])
                        except (TypeError, ValueError):
                            continue
                        xs.append(x)
                        ys.append(y)
                    if len(xs) > 2:
                        per_image[m] = {"pearson": pearson(xs, ys),
                                        "spearman": spearman(xs, ys), "n": len(xs)}
                print(f"\nPer-image correlation (n={len(paired)}):")
                for m, v in per_image.items():
                    print(f"  {m:<10} pearson={v.get('pearson')}  spearman={v.get('spearman')}")

    out = {
        "judge_model": judge_model,
        "note": "Across-configuration correlations use the paper's reported per-experiment "
                "automatic metrics; per-image correlations require --metrics-csv.",
        "cross_config_table": table,
        "per_image": per_image,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()