# Arabic zero-shot VLM prompt ablation

The existing Arabic runs use a standard one-sentence COCO-style instruction.
This experiment adds two fixed alternatives while keeping the image set, VLM,
temperature, output language, and Google-MT Arabic references unchanged:

- `concise`: one Modern Standard Arabic sentence with at most 12 words;
- `grounded`: one Modern Standard Arabic sentence that explicitly avoids
  guesses and visually unsupported objects or relations.

The baseline is already available as
`vlm_caption__<model>__ar__metrics__ar_google.json`; the runner only generates
the two new variants.

```bash
source .venv-caption-metrics/bin/activate
bash analysis/scripts/run_arabic_prompt_ablation.sh --no-meteor
```

For a smoke test use `--limit 20`. Each generated output is stored separately
under `analysis/results/vlm_caption__<model>__ar__<condition>.json`; matching
COCO-metric files are written beside it.
