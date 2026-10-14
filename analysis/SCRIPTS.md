# LLM Improvement Scripts — Documentation

Local Ollama LLMs (GPU workstation) used to improve the Arabic image-captioning
paper **without retraining**: they analyze the already-generated captions and the
MT-produced Arabic reference data.

Two data modes are supported:
- `--data-dir DIR` — the notebook's `.sav` caption pickles (generated captions,
  Google/IBM references). These live only in the private
  `basem-ahmed/image_captioning_paper` repo; not present on this machine.
- `--xlsx` — the paper's own workbook `Image Captioning for The Arabic
  Language.xlsx` (Sheet5: 1,200 images, 6,006 captions, English reference +
  Google-MT Arabic). This is the **full dataset** mode used for the paper-ready
  results below; it evaluates the MT-produced Arabic resource itself.

## Scripts Overview

| # | Script | Suggestion | Purpose | Input | Output |
|---|--------|-----------|---------|-------|--------|
| 1 | `scripts/llm_judge.py` | 1 | LLM-as-Judge semantic evaluation: scores captions (0–5) on fluency, relevance, object-accuracy, overall against references | `.sav` pickles, `--xlsx`, or built-in sample | `results/llm_judge_results.json`, `results/llm_judge_report.txt` |
| 1b | `scripts/llm_correlation.py` | 1b | Correlates LLM-judge scores with the paper's automatic metrics (cross-config) OR with per-caption back-translate fidelity (`--backtranslate-json`) | `results/llm_judge_results.json`, optional `--metrics-csv` / `--backtranslate-json` | `results/llm_correlation.json` |
| 2 | `scripts/llm_error_analysis.py` | 2 | Categorizes failures: hallucination, Arabic grammar errors, MT artifacts (translationese), repetition; + statistical repetition ratio | `.sav` pickles, `--xlsx`, or built-in sample | `results/llm_error_analysis.json`, `results/llm_error_analysis_report.txt` |
| 3 | `scripts/llm_backtranslate.py` | 3 | LLM round-trip back-translation (Arabic→English) fidelity, via embedding cosine, token F1, BLEU-1 | `.sav` pickles, `--xlsx`, or built-in sample | `results/llm_backtranslate_results.json`, `results/llm_backtranslate_report.txt` |
| — | `scripts/ollama_helper.py` | shared | Ollama API helper: host normalization, model availability, `think:false` for qwen3, JSON extraction (+ truncated-JSON retry), embeddings | — | — |
| — | `scripts/data_loader.py` | shared | Loads `.sav` pickles, the full xlsx workbook (`load_xlsx_rows`), or built-in sample (paper Table 3) | `.sav` / xlsx | dict of caption sets / records |

## Usage

```bash
# Full dataset mode (used for the paper's LLM-evaluation section):
# 1. LLM-as-Judge over all 6,006 captions
python3 scripts/llm_judge.py --model qwen3:8b --xlsx

# 2. Error / qualitative analysis over all 6,006 captions
python3 scripts/llm_error_analysis.py --model qwen3:8b --xlsx

# 3. Round-trip back-translation fidelity (up to 2 MT captions per image)
python3 scripts/llm_backtranslate.py --model qwen3:8b --xlsx

# 1b. Per-caption correlation: judge scores vs round-trip fidelity
python3 scripts/llm_correlation.py \
    --judge-json results/llm_judge_results.json \
    --backtranslate-json results/llm_backtranslate_results.json

# .sav artifact mode (when the pickles are available):
python3 scripts/llm_judge.py --model qwen3:8b --limit 20 --data-dir /path/to/sav/files
python3 scripts/llm_correlation.py
```

`--data-dir` points at the notebook's `.sav` artifacts (from
`basem-ahmed/image_captioning_paper`). Without them, `--xlsx` gives the full
dataset and scripts fall back to a small built-in sample only if neither is
given.

## Model Options

`ollama list` shows many candidate models. Recommended:

| Role | Model | Why |
|------|-------|-----|
| Judge / analysis | `qwen3:8b` (default) | fast, good Arabic, supports `think:false` |
| High-quality judge | `qwen3:32b`, `gemma3:27b`, `cogito:32b` | stronger semantic judgment |
| Also supported | `qwen3.8:latest` | user-requested alternative |
| Embeddings | `all-minilm:latest` (default) | fast, small; `mxbai-embed-large`, `bge-m3` also installed |

## Results (full dataset — verified on the GPU workstation)

Run with `qwen3:8b` (judge) + `all-minilm:latest` (embeddings) on the full xlsx
dataset: **1,200 images, 6,006 Google-MT Arabic captions**. These numbers are
used in the paper's new §LLM-Based Evaluation of Machine-Translated Captions.

| Script | Full-dataset result (see `results/` for details) |
|--------|--------------------------------------------------|
| `llm_judge.py` | ar_google: fluency 4.96, relevance 4.40, obj-acc 4.13, **overall 4.35** (n=6,006). The MT Arabic resource reads as fluent MSA and is judged usable |
| `llm_error_analysis.py` | 15,505 issues total: **MT artifacts 6,412** (~107% of captions), **hallucination 5,254**, **grammar 3,611**, repetition 226; mean repetition ratio 0.012; LLM mean overall quality 2.61/5 |
| `llm_backtranslate.py` | ar_google (n=2,400): semantic_sim 0.72, token-F1 0.54, **BLEU-1 0.69** round-trip fidelity |
| `llm_correlation.py` | Per-caption Pearson (n=2,416) judge × fidelity: BLEU-1 ↔ overall r=0.22, BLEU-1 ↔ obj-acc r=0.24; semantic_sim ↔ overall r=0.20. Positive but modest — judge and fidelity agree |

Notes:
- Earlier demo results (built-in 3-record sample) are superseded by the above
  full-dataset numbers; the sample had over-represented hallucination examples.
- The 5 error-analysis items that hit transient Ollama connection errors are
  recorded with empty `issues` (see `"error"` field); they do not affect the
  category counts materially.

## Files

```
llm_improvements/
├── SELECTED_SUGGESTIONS.md   # plan & implementation notes
├── SCRIPTS.md                # this file
├── scripts/
│   ├── ollama_helper.py
│   ├── data_loader.py
│   ├── llm_judge.py
│   ├── llm_correlation.py
│   ├── llm_error_analysis.py
│   └── llm_backtranslate.py
├── outputs/                  # full-run logs
│   ├── judge_full.log
│   ├── error_full.log
│   └── backtranslate_full.log
└── results/
    ├── llm_judge_results.json / _report.txt      (full: 6,006 items)
    ├── llm_correlation.json                       (full: per-caption, n=2,416)
    ├── llm_error_analysis.json / _report.txt      (full: 6,006 items)
    ├── llm_backtranslate_results.json / _report.txt (full: 2,400 items)
```