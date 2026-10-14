#!/usr/bin/env python3
"""Experiments V1/V2 - zero-shot captioning with vision LLMs (Ollama).

V1 (--lang en): the vision LLM writes a one-sentence English caption for each
    COCO test image. Evaluated against the English references to benchmark
    directly against the paper's Exp1 trained-model baseline.
V2 (--lang ar): the vision LLM writes a one-sentence Modern Standard Arabic
    caption for each image. Evaluated against the Google-MT references and the
    manual Arabic references (direct competition to the MT-pipeline premise).

Usage:
    python3 vlm_caption.py --model qwen3.8:27b --lang en --image-dir DIR
    python3 vlm_caption.py --model gemma4:12b --lang ar --use-venv  # pycocoevalcap
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

PROMPTS = {
    "en": ("Describe the content of this image in exactly one fluent English "
           "sentence, in the style of an MS COCO image caption. Output only the "
           "caption sentence, with no commentary."),
    "ar": ("صف محتوى هذه الصورة في جملة عربية فصيحة واحدة بأسلوب وصف الصور "
           "الشائع في مجموعة بيانات كوكو. أخرج الجملة العربية فقط دون أي تعليق أو "
           "ترجمة."),
}


def load_images(limit, image_dir, keys):
    rows = []
    missing = []
    for k in keys:
        p = os.path.join(image_dir, k)
        if os.path.exists(p):
            rows.append((k, p))
            if limit and len(rows) >= limit:
                break
        else:
            missing.append(k)
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="vision-capable Ollama model, e.g. gemma4:12b, qwen3.8:27b")
    ap.add_argument("--lang", choices=["en", "ar"], default="en")
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--image-dir", default=DEFAULT_IMAGES)
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--prompt", default=None,
                    help="override the built-in language prompt for an ablation")
    ap.add_argument("--run-tag", default=None,
                    help="suffix output names for an ablation, e.g. grounded")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    host = normalize_host(args.host)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = load_xlsx_rows(args.xlsx)
    keys = [r["key"] for r in dataset]
    img_rows, missing = load_images(args.limit, args.image_dir, keys)
    if missing:
        print(f"WARNING: {len(missing)} images missing from {args.image_dir}; "
              f"first few: {missing[:5]}")
    if not img_rows:
        raise SystemExit(f"No images found in {args.image_dir}. "
                         "Run download_coco.py first.")
    by_key = {r["key"]: r for r in dataset}

    print(f"Captioning {len(img_rows)} image(s) with '{args.model}' "
          f"(lang={args.lang})...")
    tag = slug_model(args.model)
    run_suffix = f"__{slug_model(args.run_tag)}" if args.run_tag else ""
    out_json = os.path.join(RESULTS_DIR, f"vlm_caption__{tag}__{args.lang}{run_suffix}.json")
    out_txt = os.path.join(RESULTS_DIR, f"vlm_caption__{tag}__{args.lang}{run_suffix}.txt")
    prompt = args.prompt or PROMPTS[args.lang]

    per_item = []
    for i, (key, img) in enumerate(img_rows, 1):
        rec = by_key[key]
        try:
            caption = generate_vision(prompt, model=args.model,
                                      images=[img], host=host,
                                      temperature=args.temperature,
                                      num_predict=140)
        except Exception as e:
            caption = None
        per_item.append({
            "key": key,
            "caption": caption,
            "eng_refs": rec.get("eng_ref", []),
            "ar_google": rec.get("ar_google", []),
            "ar_manual": rec.get("ar_manual", []),
            "error": None if caption else str(e),
        })
        if args.verbose:
            print(f"[{i}/{len(img_rows)}] {key}: {caption!r}")
        if i % 100 == 0:
            print(f"  [{i}/{len(img_rows)}] done", flush=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "lang": args.lang, "run_tag": args.run_tag,
                   "prompt": prompt, "n": len(per_item),
                   "per_item": per_item}, f, ensure_ascii=False, indent=2)

    ok = [x for x in per_item if x.get("caption")]
    lines = [f"VLM captioning (model: {args.model}, lang: {args.lang})",
             "=" * 60,
             f"images: {len(per_item)}   captioning success: {len(ok)}"]
    if args.lang == "en" and ok:
        lens = [len(x["caption"].split()) for x in ok]
        lines.append(f"mean caption length (words): {statistics.mean(lens):.1f}")
    elif args.lang == "ar" and ok:
        lens = [len(x["caption"].replace(' ', '')) for x in ok]
        lines.append(f"mean caption length (chars): {statistics.mean(lens):.1f}")
    report = "\n".join(lines)
    print("\n" + report)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved JSON -> {out_json}\nSaved report -> {out_txt}")


if __name__ == "__main__":
    main()
