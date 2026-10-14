# Direct Arabic VLM vs translated English VLM baseline

This experiment is the modern zero-shot counterpart of the historical
``translated English output`` condition. For each VLM, it:

1. uses the stored direct English VLM caption for every test image;
2. translates that English caption to Modern Standard Arabic with a local
   Ollama translator (default: `ministral-3:14b`);
3. evaluates the translated Arabic caption against the same Google-MT Arabic
   references used for the direct Arabic VLM output.

Activate the metrics environment and run:

```bash
source .venv-caption-metrics/bin/activate
bash analysis/scripts/run_translated_vlm_baseline.sh --no-meteor
```

For a smoke test, add `--limit 20`. The translation JSON checkpoints after each
caption; rerunning the same command resumes interrupted translation work.

Compare each resulting
`vlm_caption__<model>__en_translated_ar__metrics__ar_google.json` with the
existing direct-Arabic file
`vlm_caption__<model>__ar__metrics__ar_google.json`. The comparison isolates
the generation strategy while holding the image, VLM, Arabic reference set, and
metric implementation fixed.
