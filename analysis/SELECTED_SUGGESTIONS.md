# LLM-Based Improvements to the Paper

**Goal:** Improve the paper *without* re-running model training. We leverage the workstation's
local Ollama LLMs (`qwen3:8b`, `qwen3:32b`, `gemma3:27b`, `cogito:32b`, `cogito:14b`, `qwen3-coder:30b`,
etc.) on the available GPU to analyze the *existing* generated captions and translation artifacts.

All changes use the already-trained models and already-computed metrics; nothing here retrains.

---

## Selected Suggestions (to be implemented)

### Suggestion 1 — LLM-as-Judge Semantic Evaluation

The paper currently relies only on n-gram metrics (BLEU/ROUGE/METEOR/CIDEr), which are known to
correlate weakly with human judgment, especially for morphologically rich Arabic. We add a
**semantic, reference-free LLM evaluation** of the already-generated captions.

- Feed an image's **reference captions** and the **model's generated caption** (for each experiment
  variant, both Google-MT and IBM-MT derived) to an LLM.
- The LLM scores each generated caption on a rubric: **Fluency**, **Relevance/Semantic Correctness**,
  **Object Accuracy (hallucination checking)**, and an **Overall score (0–5)**.
- Models evaluated: **qwen3:8b**, plus the larger judge models we can load.
- Output per-experiment mean scores and, crucially, the **correlation** between LLM-judge scores and
  the existing automatic metrics. This directly addresses the paper's stated *future work*
  (`use human evaluation`) as a low-cost, reproducible surrogate.
- New contributions: a quantitative semantic evaluation that can rank Exp2 vs Exp3, Google vs IBM
  more meaningfully than n-gram overlap alone.

### Suggestion 2 — LLM-Ranked Error / Qualitative Analysis

Take the *already-generated* captions and have an LLM analyze failure patterns:

- Detect **hallucinated objects** (objects in the caption not supported by the image/references),
- **grammatical / morphological errors** in Arabic,
- **MT artifacts** (translationese, e.g. wrong word choices carried from Google/IBM MT),
- Repeated / degenerate caption patterns (the model collapsing to a few templates).

Produce a categorized, citable qualitative analysis that strengthens the Discussion section with
concrete, LLM-verified examples rather than hand-waving.

### Suggestion 3 — LLM Round-Trip (Back-Translation) Quality Check

Independent of the captioning models, measure the **translation quality** of the MT-produced Arabic
captions themselves using LLM **back-translation**:

- Take the Arabic captions (Google-MT / IBM-MT produced) used as references.
- Use an LLM to translate each Arabic caption **back to English**.
- Compare the round-tripped English against the original English caption using standard MT/overlap
  metrics (e.g. a reference-based BLEU / similarity).
- This gives a **translation-fidelity signal** (how much information the MT preserved), orthogonal to
  the downstream captioning metrics, and lets us rank Google vs IBM vs LLM translation quality.

> Note on data availability: the raw caption `.sav` pickle artifacts produced by the notebook live in
> the separate `basem-ahmed/image_captioning_paper` repo (cloned in Colab), **not** in this repo.
> The scripts are written to accept whatever caption files are present. Where artifacts are missing,
> the scripts run against representative sample captions from the paper itself (e.g. Table 3) so the
> pipeline is demonstrable end-to-end.

---

## Implementation Status

- [x] Suggestion 1 — LLM-as-Judge evaluation (`scripts/llm_judge.py`)
- [x] Suggestion 1b — Correlation analysis vs automatic metrics (`scripts/llm_correlation.py`)
- [x] Suggestion 2 — LLM error/qualitative analysis (`scripts/llm_error_analysis.py`)
- [x] Suggestion 3 — LLM round-trip translation quality (`scripts/llm_backtranslate.py`)
- [x] Shared model helper (`scripts/ollama_helper.py`)

## How to Run

All scripts talk to a local Ollama server on the GPU workstation (default model
`qwen3:8b`; override with `--model`, e.g. `qwen3:32b` or `gemma3:27b` for higher
judge quality, or `qwen3.8:latest`).

```bash
# 1. LLM-as-Judge semantic evaluation of already-generated captions
python3 scripts/llm_judge.py --model qwen3:8b --limit 20 --data-dir /path/to/sav/files

# 1b. Correlate LLM scores with the paper's automatic metrics
python3 scripts/llm_correlation.py

# 2. LLM error/qualitative analysis (hallucination, grammar, MT artifacts)
python3 scripts/llm_error_analysis.py --model qwen3:8b

# 3. LLM round-trip back-translation fidelity of Google vs IBM MT captions
python3 scripts/llm_backtranslate.py --model qwen3:8b
```

Outputs are written to `results/*.json` and `results/*_report.txt`.

## Results (demo run, sample captions from paper Table 3)

All four scripts were verified end-to-end with `qwen3:8b` on the GPU workstation.
The demo uses the built-in sample captions (the `.sav` artifacts are not present
in this repo); re-run with `--data-dir` pointing at the notebook's pickles to
produce paper-ready numbers.

| Output | Content |
|--------|---------|
| `results/llm_judge_results.json` / `.txt` | Per-config semantic scores (fluency, relevance, object accuracy, overall) |
| `results/llm_correlation.json` | Pearson correlation matrix: LLM scores vs BLEU/CIDEr/METEOR/ROUGE (across configs) |
| `results/llm_error_analysis.json` / `.txt` | Categorized hallucination/grammar/MT-artifact issues with quoted spans |
| `results/llm_backtranslate_results.json` / `.txt` | Round-trip fidelity scores (embedding cosine, token F1, BLEU-1) per MT source |

### What the demo shows (illustrative, verify on full data)

- `llm_judge` ranked IBM-derived captions above Google-derived ones on the sample,
  and, notably, judged *fluency* of `ar_google`/`ar_ibm` high while *relevance/
  object_accuracy* low — evidence that n-gram metrics alone miss semantic quality.
- `llm_correlation` found BLEU-1 tracking LLM relevance/overall best, while
  METEOR correlated *negatively* with LLM semantic scores — a cautionary finding
  for the paper's reliance on METEOR stability.
- `llm_error_analysis` surfaced concrete hallucinations (e.g. "امرأة" = woman in
  an image of men), grammar errors (e.g. "موقف للسيارات" → "موقف السيارات"),
  and translationese — directly usable examples for the Discussion section.
- `llm_backtranslate` measured MTP fidelity independently (embedding cosine,
  token F1, BLEU-1), giving an MT-quality signal orthogonal to captioning metrics.
