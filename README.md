# Data and code for *Machine-Translated Resources for Image Captioning*

This repository contains the reproducibility materials for the manuscript
*Machine-Translated Resources for Image Captioning: Downstream Evidence and a
Reproducible VLM Reassessment*.

## Contents

- `data/arabic_captions.xlsx` — the 1,200-image, 6,006-caption English and
  Google-MT Arabic workbook used by the analysis scripts.
- `notebook/image_captioning_paper.ipynb` — the archived TensorFlow
  encoder--decoder training protocol from the earlier study.
- `analysis/scripts/` — executable Python and shell scripts for the VLM and
  LLM analyses reported in the manuscript.
- `analysis/results/` — machine-readable outputs and reports for zero-shot
  captioning, LLM-as-judge, image-grounded judging, round-trip fidelity,
  cross-lingual consistency, reference-style sensitivity, and prompt
  ablations.
- `analysis/*.md` — experiment descriptions and invocation notes.
- [`PROMPTS.md`](PROMPTS.md) — the exact prompt templates used for every
  reported VLM and LLM generation or judgment task.

## Reproducing the follow-up analyses

The VLM/LLM scripts use a local Ollama service.  Create an isolated Python
environment, install the dependencies, and follow the detailed commands in
[`analysis/FOLLOWUP_EXPERIMENTS.md`](analysis/FOLLOWUP_EXPERIMENTS.md) and
[`analysis/SCRIPTS.md`](analysis/SCRIPTS.md).

```bash
python3 -m venv .venv-caption-metrics
source .venv-caption-metrics/bin/activate
pip install -r requirements.txt
bash analysis/scripts/run_followup_experiments.sh --no-meteor
```

The exact local Ollama model tags and decoding settings used for the stored
outputs are documented in the manuscript.  The result files retain inputs and
outputs where applicable; model digests and random seeds were not retained by
the original runs, so regenerated generations can differ.

## Data provenance and access

The workbook contains English MS COCO captions paired with Google-MT Arabic
captions for the study's held-out images.  MS COCO images and the complete
original caption distribution are **not** included here; obtain them from the
[official MS COCO website](https://cocodataset.org/#download) and comply with
its terms.  The original model checkpoints, fixed train/test split, generated
captions from the earlier encoder--decoder experiments, and manually translated
reference subset are unavailable.

## Citation

Please cite the accompanying manuscript and the original study:

```bibtex
@inproceedings{ahmed2021mtcaptioning,
  author = {Ahmed, Basem H. and Saad, Motaz},
  title = {The Use of Machine Translation to Provide Resources for
           Under-Resourced Languages: Image Captioning Task},
  booktitle = {PICICT},
  year = {2021},
  pages = {25--29},
  doi = {10.1109/PICICT53635.2021.00017}
}
```

No license is asserted for third-party material.  Users are responsible for
respecting the terms that apply to MS COCO, machine-translation outputs, and
the referenced model software.
