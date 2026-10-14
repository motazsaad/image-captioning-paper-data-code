#!/usr/bin/env python3
"""Suggestion 3 - LLM Round-Trip (Back-Translation) Translation Quality.

The paper's core premise is that MT services (Google, IBM) can create usable
Arabic resources. This script independently measures the *fidelity* of those MT
Arabic captions, without any captioning model: we back-translate the Arabic
captions to English with a local LLM and compare the round-trip English against
the original English caption.

Scores per caption:
  - ``semantic_sim``  cosine similarity of embeddings (LLM/embedding model)
  - ``token_f1``      F1 of token overlap, dependency-free
  - ``bleu1``         unigram precision w/ brevity penalty (self-contained impl)

These give a translation-fidelity ranking of Google MT vs IBM MT (and can be
extended to LLM-translated data) that feeds directly into the paper's analysis.

Usage:
    python3 llm_backtranslate.py --model qwen3:8b --limit 10 [--data-dir DIR]
"""
import argparse
import json
import math
import os
import re
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_helper import (generate, normalize_host, embed, cosine,  # noqa: E402
                           slug_model)
from data_loader import (load_caption_file, sample_dataset,        # noqa: E402
                         load_xlsx_rows)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT_JSON = os.path.join(RESULTS_DIR, "llm_backtranslate_results.json")
OUT_TXT = os.path.join(RESULTS_DIR, "llm_backtranslate_report.txt")

EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "all-minilm:latest")

BACKTRANS_PROMPT = (
    "Back-translate the following Arabic image caption to natural, fluent "
    "English. Output ONLY the English translation with no commentary:\n"
    "{arabic}"
)


def _tokens(text):
    return [t.lower() for t in re.split(r"[^A-Za-z0-9']+", text) if t]


def token_f1(hyp, ref):
    h, r = collections.Counter(_tokens(hyp)), collections.Counter(_tokens(ref))
    inter = sum((h & r).values())
    total = sum(h.values()) + sum(r.values())
    return (2 * inter / total) if total else 0.0


def bleu1(hyp, refs):
    """Self-contained BLEU-1 with brevity penalty (no external deps)."""
    h = _tokens(hyp)
    best = 0.0
    for ref in refs:
        r = _tokens(ref)
        if not r or not h:
            continue
        hits = sum(min(collections.Counter(h).get(w, 0), r.count(w)) for w in set(h))
        prec = hits / len(h)
        brevity = math.exp(1 - len(r) / len(h)) if len(h) < len(r) else 1.0
        best = max(best, prec * brevity)
    return best


def build_inputs(limit, data_dir, xlsx=False, xlsx_path=None):
    """Return list of records: {key, eng_refs, ar_google, ar_ibm}."""
    if xlsx:
        rows = []
        for rec in load_xlsx_rows(xlsx_path):
            row = {"key": rec["key"], "eng_refs": rec.get("eng_ref", []),
                   "ar_google": rec.get("ar_google", []), "ar_ibm": rec.get("ar_manual", [])}
            rows.append(row)
        if limit > 0:
            rows = rows[:limit]
        return rows

    eng_ref = load_caption_file("image_path_to_caption_en.sav", [data_dir] if data_dir else None)
    ar_google = load_caption_file("image_path_to_caption_ar.sav", [data_dir] if data_dir else None)
    ar_ibm = load_caption_file("image_path_to_caption_ar_ibm.sav", [data_dir] if data_dir else None)

    if eng_ref:
        keys = list(eng_ref.keys())
        if limit > 0:
            keys = keys[:limit]
        rows = []
        for k in keys:
            row = {"key": k, "eng_refs": eng_ref[k], "ar_google": None, "ar_ibm": None}
            if ar_google:
                row["ar_google"] = ar_google.get(k, [])
            if ar_ibm:
                row["ar_ibm"] = ar_ibm.get(k, [])
            rows.append(row)
        return rows

    # Fallback: built-in sample.
    rows = []
    for key, rec in sample_dataset().items():
        rows.append({"key": key,
                     "eng_refs": rec.get("eng_ref", []),
                     "ar_google": rec.get("ar_google_ref", []),
                     "ar_ibm": rec.get("ar_ibm_ref", [])})
    if limit > 0:
        rows = rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--embed-model", default=None,
                    help=f"embedding model for cosine similarity (default: "
                         f"$OLLAMA_EMBED_MODEL or 'all-minilm:latest')")
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--xlsx", action="store_true",
                    help="read the full dataset from the paper's Excel workbook")
    ap.add_argument("--xlsx-path", default=None,
                    help="explicit path to the Excel workbook (defaults to data/arabic_captions.xlsx)")
    args = ap.parse_args()

    host = normalize_host(args.host)
    embed_model = args.embed_model or os.environ.get("OLLAMA_EMBED_MODEL", EMBED_MODEL)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    inputs = build_inputs(args.limit, args.data_dir, args.xlsx, args.xlsx_path)
    print(f"Back-translating {len(inputs)} Arabic reference caption(s) "
          f"with '{args.model}'...")

    per_item, by_source = [], collections.defaultdict(list)
    for i, row in enumerate(inputs, 1):
        print(f"[{i}/{len(inputs)}] key={row['key']}")
        for source in ("ar_google", "ar_ibm"):
            arabics = [c for c in (row.get(source) or []) if c]
            if not arabics:
                continue
            for ar in arabics[:2]:   # cap per-image work on the demo set
                eng_ref = (row["eng_refs"] or [""])[0]
                rec = {
                    "key": row["key"], "source": source, "arabic": ar,
                    "backtranslated_en": None, "original_en": eng_ref,
                    "semantic_sim": None, "token_f1": 0.0, "bleu1": 0.0,
                    "error": None,
                }
                try:
                    back_en = generate(BACKTRANS_PROMPT.format(arabic=ar),
                                       model=args.model, host=host,
                                       num_predict=120)
                    sim = cosine(embed(back_en, model=embed_model, host=host),
                                 embed(eng_ref, model=embed_model, host=host)) \
                        if back_en and eng_ref else None
                    rec.update({
                        "backtranslated_en": back_en,
                        "semantic_sim": round(sim, 4) if sim is not None else None,
                        "token_f1": round(token_f1(back_en, eng_ref), 4) if eng_ref else 0.0,
                        "bleu1": round(bleu1(back_en, row["eng_refs"]), 4)
                                 if row["eng_refs"] else 0.0,
                    })
                except Exception as e:
                    # A stalled/failed Ollama request must skip the caption,
                    # not freeze or crash the whole run.
                    rec["error"] = str(e)
                per_item.append(rec)
                by_source[source].append(rec)
                if args.verbose and rec["backtranslated_en"]:
                    print(f"   [{source}] AR: {ar}")
                    print(f"            EN: {rec['backtranslated_en']!r}")

    summary = {}
    for src, items in by_source.items():
        n = len(items)
        summary[src] = {
            "n": n,
            "mean_semantic_sim": round(sum(x["semantic_sim"] or 0 for x in items) / n, 4),
            "mean_token_f1": round(sum(x["token_f1"] for x in items) / n, 4),
            "mean_bleu1": round(sum(x["bleu1"] for x in items) / n, 4),
        }

    tag = slug_model(args.model) + "__" + slug_model(embed_model)
    out_json = os.path.join(RESULTS_DIR, f"llm_backtranslate_results__{tag}.json")
    out_txt = os.path.join(RESULTS_DIR, f"llm_backtranslate_report__{tag}.txt")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "embed_model": embed_model,
                   "per_item": per_item, "summary": summary},
                  f, ensure_ascii=False, indent=2)

    lines = [f"LLM Round-Trip Translation Fidelity (model: {args.model}, "
             f"embed: {embed_model})", "=" * 60,
             f"{'source':<12}{'n':<5}{'sem_sim':<10}{'token_F1':<10}{'BLEU-1':<8}"]
    for src, s in summary.items():
        lines.append(f"{src:<12}{s['n']:<5}{s['mean_semantic_sim']:<10}"
                     f"{s['mean_token_f1']:<10}{s['mean_bleu1']:<8}")
    report = "\n".join(lines)
    print("\n" + report)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    # Backward-compatible aliases (single default-model run behaviour).
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "embed_model": embed_model,
                   "per_item": per_item, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved JSON -> {out_json}\nSaved report -> {out_txt}")


if __name__ == "__main__":
    main()