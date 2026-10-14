#!/usr/bin/env python3
"""Load caption artifacts produced by the paper's notebook.

The training pipeline dumped several pickle files (see image_captioning_paper.ipynb):

    image_path_to_caption_en.sav          English reference captions
    image_path_to_caption_ar.sav          Google-MT Arabic reference captions
    image_path_to_caption_ar_ibm.sav      IBM-MT Arabic reference captions
    image_path_to_caption_gen_en.sav      Generated English captions (Exp1 hypothesis)
    image_path_to_caption_gen_en_ar.sav   English output translated to Arabic (Exp3/5)
    image_path_to_caption_gen_ar.sav      Generated Arabic captions (Exp2/4 hypothesis)

Each file maps `{image_id_or_path: [caption, ...]}` and captions carry <start>/<end>
markers. These files live in the separate basem-ahmed/image_captioning_paper repo
(cloned in Colab) and may not be present here, so we also provide a small built-in
sample derived from the paper's own Table 3 so every script is runnable end-to-end.
"""
import os
import glob
import pickle
from collections import defaultdict


CAPTION_FILES = {
    "eng_ref": "image_path_to_caption_en.sav",
    "ar_google_ref": "image_path_to_caption_ar.sav",
    "ar_ibm_ref": "image_path_to_caption_ar_ibm.sav",
    "gen_en": "image_path_to_caption_gen_en.sav",
    "gen_en_ar": "image_path_to_caption_gen_en_ar.sav",
    "gen_ar": "image_path_to_caption_gen_ar.sav",
}

_IMAGE_ROOTS = (
    "/content/train2014/",
    "/content/image_captioning_paper/",
    "/content/",
    "",
)


def _normalise(caption):
    """Strip <start>/<end> markers and collapse whitespace."""
    c = caption.replace("<start>", "").replace("<end>", "").strip()
    return " ".join(c.split())


def _find(filename, roots):
    for root in roots:
        path = os.path.join(root, filename)
        if os.path.exists(path):
            return path
    # search recursively under the provided roots
    for root in roots:
        if root and os.path.isdir(root):
            hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
            if hits:
                return hits[0]
    return None


def _is_coco_id(key):
    """Keys in the gen files are sometimes bare COCO ids like
    'COCO_train2014_000000073342.jpg'."""
    return isinstance(key, str) and key.lower().startswith("coco")


def load_caption_file(filename, search_dirs=None):
    """Return dict {display_key: [clean_caption, ...]} for a single file.

    Keys are normalised to the basename of the image path so different pickle
    files (which may key by full path vs. bare id) line up with each other.
    """
    roots = list(search_dirs or [])
    roots += [d for d in _IMAGE_ROOTS if d not in roots]
    path = _find(filename, roots)
    if not path:
        return None

    with open(path, "rb") as f:
        raw = pickle.load(f)

    out = {}
    for key, caps in raw.items():
        if isinstance(caps, str):
            caps = [caps]
        disp = os.path.basename(key) if isinstance(key, str) else str(key)
        out[disp] = [_normalise(c) for c in caps if c]
    return out


def describe_available(search_dirs=None):
    """Report which artifact files are present and their size."""
    result = {}
    for role, filename in CAPTION_FILES.items():
        data = load_caption_file(filename, search_dirs)
        if data is not None:
            result[role] = (filename, len(data))
    return result


# ---------------------------------------------------------------------------
# Built-in sample data (paper Table 3) so scripts always have something to run.
# Format: {key: {"img": <caption>, "ar_google": [...], ...}}
# ---------------------------------------------------------------------------
SAMPLE = {
    "sample_1": {
        "img": "image of two men standing near motorcycles parked inside a tent",
        "eng_ref": [
            "a couple of motorcycles are parked in a tent",
            "two men stand near motorcycles parked in a tent",
            "two men with motorcycles stand under a tent",
            "two men talking in a tent with motorbikes in it",
        ],
        "ar_google_ref": [
            "حمام أبيض صغير مع العديد من المناشف البيضاء",
            "حمام صغير جداً يحتوي على بعض المناشف على الأرض",
            "حمام صغير مع مجموعة من المناشف فيه",
        ],
        "ar_ibm_ref": [
            "واقفة اثنين من الدراجات النارية في خيمة",
            "رجلان يقفان قرب دراجة نارية متوقفة في خيمة",
            "رجلان يركبان دراجتين ناريتين يقفان تحت خيمة",
        ],
        "gen_en": "a man holding a skateboard while standing in his hands",
        "gen_en_ar": "رجل يرتدي ربطة عنق يتحدث على هاتفه",
        "gen_ar_google": "مرحاض مرحاض مع بطاقة مع بالوعة في الحمام",
        "gen_ar_ibm": "امرأة تقف في موقف للسيارات بالقرب من شاحنة",
    },
    "sample_2": {
        "img": "image of several zebras standing close together outdoors",
        "eng_ref": [
            "a herd of zebra standing next to each other",
            "a closeup of the stripes of a herd of zebras",
            "a closeup shot of several zebras standing together",
        ],
        "ar_google_ref": [
            "قطيع من الحمار الوحشي يقف بجانب بعضهما البعض",
            "صورة مقربة لخطوط قطيع من الحمر الوحشية",
            "لقطة مقربة للعديد من الحمر الوحشية تقف معاً",
        ],
        "ar_ibm_ref": [
            "قطيع من الحمر الوحشية يقف بجانب بعضهما البعض",
            "لقطة مقربة لعدة حمر وحشية تقف معاً",
        ],
        "gen_en": "a zebra out on tall grass near a post turning toward the camera",
        "gen_en_ar": "صف من الحمر الوحشية التي تواجه بعضها البعض",
        "gen_ar_google": "حمار وحشي يقف في العراء بعيداً عن الكاميرا",
        "gen_ar_ibm": "عدة حمر وحشية تقف في العراء",
    },
    "sample_3": {
        "img": "image of a man smiling and talking on a cell phone while wearing a hat",
        "eng_ref": [
            "a man smiles as he talks on a cell phone",
            "a man is talking on a cellphone outside",
            "a man talking on a phone while wearing a hat",
            "a man talking on a cell phone in a stocking cap",
        ],
        "ar_google_ref": [
            "رجل يبتسم وهو يتحدث على الهاتف الخليوي",
            "رجل يتحدث على هاتف محمول بالخارج",
            "رجل يتحدث على الهاتف وهو يرتدي قبعة",
        ],
        "ar_ibm_ref": [
            "رجل يتحدث على هاتف خلوي في الخارج",
            "رجل يرتدي قبعة ويتحدث على الهاتف",
        ],
        "gen_en": "a man holding a skateboard while standing",
        "gen_en_ar": "رجل يرتدي ربطة عنق يتحدث على هاتفه",
        "gen_ar_google": "رجل يتحدث على الهاتف وهو يرتدي قبعة شتوية",
        "gen_ar_ibm": "رجل يرتدي قبعة ويتحدث على هاتفه",
    },
}


def sample_dataset():
    """Return the built-in sample as {key: record} for read-only demos."""
    return {k: dict(v) for k, v in SAMPLE.items()}


# ---------------------------------------------------------------------------
# XLSX-based full-dataset loader (the paper's own workbook).
# ---------------------------------------------------------------------------
def load_xlsx_rows(xlsx_path=None, group_by_image=True):
    """Read the paper's workbook into evaluation records.

    Sheet5 has the full MS COCO validation/test set: one row per (image,
    English caption, Google-MT Arabic translation, [Manual Arabic]). Because
    the ``Manual Translation`` column is populated on only a tiny subset in the
    workbook, the default source is the Google-MT Arabic translation evaluated
    against the English caption(s) of the same image.

    Returns a list of records (one per image, or one per raw caption row if
    ``group_by_image`` is False):

        {"key": <img_id>, "eng_ref": [english captions of that image],
         "ar_google": [google arabic captions of that image],
         "ar_manual": [manual arabic ...] (may be empty),
         "img": <row image id>}
    """
    if xlsx_path is None:
        for candidate in (
            "Image Captioning for The Arabic Language.xlsx",
            os.path.join("data", "arabic_captions.xlsx"),
        ):
            cand = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), candidate)
            if os.path.exists(cand):
                xlsx_path = cand
                break
    try:
        import openpyxl
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "openpyxl is required to read the paper workbook: pip install openpyxl") from e

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Sheet5"]
    by_image = defaultdict(lambda: {"eng_ref": [], "ar_google": [], "ar_manual": []})
    for row in ws.iter_rows(values_only=True):
        if row[0] is None or row[0] in (0, "img_id"):
            continue
        img_id = str(row[0])
        eng = _normalise(row[1]) if row[1] else ""
        google = _normalise(row[2]) if row[2] else ""
        manual = _normalise(row[3]) if len(row) > 3 and row[3] else ""
        if eng:
            by_image[img_id]["eng_ref"].append(eng)
        if google:
            by_image[img_id]["ar_google"].append(google)
        if manual:
            by_image[img_id]["ar_manual"].append(manual)

    if group_by_image:
        return [{"key": k, "img": k, **v} for k, v in by_image.items()]

    rows = []
    for img_id, d in by_image.items():
        for i in range(max(len(d["eng_ref"]), len(d["ar_google"]), len(d["ar_manual"]))):
            rows.append({
                "key": img_id,
                "img": img_id,
                "eng_ref": [d["eng_ref"][i]] if i < len(d["eng_ref"]) else [],
                "ar_google": [d["ar_google"][i]] if i < len(d["ar_google"]) else [],
                "ar_manual": [d["ar_manual"][i]] if i < len(d["ar_manual"]) else [],
            })
    return rows


if __name__ == "__main__":
    print("Available artifact files (pickle):")
    info = describe_available()
    if not info:
        print("  None found on this machine (expected).")
    for role, (fname, n) in info.items():
        print(f"  - {role}: {fname} ({n} entries)")
    print(f"\nBuilt-in sample records: {len(sample_dataset())}")
    try:
        rows = load_xlsx_rows()
        print(f"\nXLSX records (by image): {len(rows)}")
        if rows:
            print("  sample:", rows[0]["key"], rows[0]["eng_ref"][:1], rows[0]["ar_google"][:1],
                  "manual:", len(rows[0]["ar_manual"]))
    except SystemExit as e:
        print(f"\nXLSX load: {e}")
