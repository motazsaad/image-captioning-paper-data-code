# Annotation-Free Follow-Up Experiments

This runnable set implements the three priority additions in
[`docs/EXPERIMENT_SUGGESTIONS.md`](../docs/EXPERIMENT_SUGGESTIONS.md). It
assumes that `run_vision.sh` has already created English and Arabic caption JSON
files for `qwen3.8:27b` and `gemma4:12b` in `analysis/results/`.

## Workstation setup

Start Ollama and make sure these local models are available (names may be
changed through environment variables):

- caption VLMs: `qwen3.8:27b`, `gemma4:12b`;
- back-translation and semantic judging: `ministral-3:14b`;
- embeddings: `qwen3-embedding:4b`.

`vlm_metrics.py` additionally requires a Python environment with
`pycocoevalcap` and Java for METEOR. Pass its Python executable in
`METRICS_PY`. If it is absent, the runner completes the GPU-based analyses and
prepares the three reference-style caption files, while skipping only the COCO
metric calculation.

```bash
cd analysis/scripts
METRICS_PY=/path/to/capcaption/bin/python ./run_followup_experiments.sh
```

For a smoke test:

```bash
./run_followup_experiments.sh --limit 20 --no-meteor
```

Useful options:

- `--skip-judge`: skip the LLM semantic-consistency score in the first
  experiment while retaining back-translation and embeddings.
- `--no-meteor`: avoid the slow Java METEOR stage.
- `--dry-run`: print planned commands only.
- `TRANSLATION_MODEL`, `JUDGE_MODEL`, and `EMBED_MODEL`: environment-variable
  overrides for local Ollama model names.

Every LLM/embedding experiment checkpoints after each caption. Re-run the
same command to resume; pass `--restart` directly to an individual Python
script only when intentionally discarding its checkpoint.

## 1. Cross-lingual VLM consistency

`crosslingual_consistency.py` pairs a VLM's English and Arabic outputs by image
key. It back-translates the Arabic caption, then produces per image:

- the original English and Arabic captions;
- the back-translated English caption;
- embedding cosine, token F1, and BLEU-1 between the direct-English and
  back-translated captions;
- an optional 0--5 LLM semantic-consistency score and short rationale.

Output: `analysis/results/crosslingual_consistency__<model>.json`.

## 2. Reference-style sensitivity

`reference_style_sensitivity.py` reads the cross-lingual output and materializes
three `vlm_metrics.py`-compatible files:

1. direct English VLM captions scored against English references;
2. original Arabic VLM captions scored against Google-MT Arabic references;
3. back-translated Arabic VLM captions scored against English references.

When `--metrics-python` is provided, it invokes `vlm_metrics.py` on all three.
Comparing these metric files separates changes caused by language/reference
style from changes in the visual content represented by the caption.

## 3. VLM disagreement as uncertainty

`vlm_disagreement.py` pairs Qwen and Gemma captions in the same language and
uses embedding distance (`1 - cosine`) as an annotation-free disagreement
score. The output ranks the highest-disagreement images and preserves both
captions, allowing focused qualitative inspection later.

Outputs: `analysis/results/vlm_disagreement__qwen3_8_27b__vs__gemma4_12b__en.json`
and the corresponding Arabic file.
