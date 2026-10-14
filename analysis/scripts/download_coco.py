#!/usr/bin/env python3
"""Download the MS COCO train2014 images referenced by the paper's test set.

The workbook ``data/arabic_captions.xlsx`` (Sheet5) lists 1,200 test images as
bare ids like ``000443.jpg`` which map to COCO 2014 train images named
``COCO_train2014_%012d.jpg``. This script fetches exactly those files from the
official COCO image CDN (``images.cocodataset.org``).

Usage:
    python3 download_coco.py [--xlsx PATH] [--out-dir DIR] [--workers N] [--limit N]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DEFAULT_XLSX = os.path.join(REPO_ROOT, "data", "arabic_captions.xlsx")
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "coco", "train2014")
BASE_URL = "http://images.cocodataset.org/train2014"


def coco_filename(img_id):
    """'000443.jpg' -> 'COCO_train2014_000000000443.jpg'"""
    stem = img_id[:-4] if img_id.lower().endswith(".jpg") else img_id
    return f"COCO_train2014_{stem.zfill(12)}.jpg"


def dataset_ids(xlsx_path):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Sheet5"]
    seen = {}
    for row in ws.iter_rows(values_only=True):
        if row[0] is None or row[0] in (0, "img_id"):
            continue
        seen[str(row[0])] = None
    return list(seen.keys())


def fetch_one(args):
    img_id, out_dir = args
    name = coco_filename(img_id)
    dest = os.path.join(out_dir, img_id)
    if os.path.exists(dest) and os.path.getsize(dest) > 5000:
        return img_id, True, "cached"
    url = f"{BASE_URL}/{name}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) > 5000:
                with open(dest, "wb") as f:
                    f.write(r.content)
                return img_id, True, f"{len(r.content)}B"
            if r.status_code == 404:
                return img_id, False, "404"
        except Exception as e:
            if attempt == 2:
                return img_id, False, str(e)
            time.sleep(2 * (attempt + 1))
    return img_id, False, "retries exhausted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 1,200")
    args = ap.parse_args()

    ids = dataset_ids(args.xlsx)
    if args.limit > 0:
        ids = ids[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Downloading {len(ids)} images from {BASE_URL} -> {args.out_dir}")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, (i, args.out_dir)): i for i in ids}
        for j, fut in enumerate(as_completed(futs), 1):
            img_id, success, info = fut.result()
            ok += success
            fail += not success
            if not success:
                print(f"  FAIL {img_id}: {info}", flush=True)
            if j % 100 == 0 or j == len(futs):
                print(f"  [{j}/{len(futs)}] ok={ok} fail={fail}", flush=True)
    print(f"DONE: ok={ok} fail={fail}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()