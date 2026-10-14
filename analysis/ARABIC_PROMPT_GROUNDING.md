# Image-grounded evaluation of Arabic prompt variants

The metric ablation found that concise prompts improve reference overlap and
grounded prompts reduce it. This follow-up checks whether the grounded prompt
is nevertheless more faithful to the actual image.

For each captioning model, an independent VLM judges the standard, concise, and
grounded Arabic captions against image pixels. It returns fluency, relevance,
object accuracy, overall quality, and a count of flagged hallucinated spans.
Cross-model judging reduces, but does not eliminate, self-evaluation bias:

- Gemma judges Qwen captions.
- Qwen judges Gemma captions.

```bash
source .venv-caption-metrics/bin/activate
bash analysis/scripts/run_arabic_prompt_judge.sh
```

Use `--limit 20` for a smoke test. Each run checkpoints after an image; rerun
the command to continue after an interruption.
