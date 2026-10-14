# Prompt catalogue

This file records the exact prompt templates used to obtain the results
reported in the accompanying manuscript.  Placeholders in braces are replaced
per item.  All calls use a single user message, disabled streaming, and
disabled model thinking.  The role-specific temperatures and output-token
limits are listed in the manuscript's model-settings table.

## Zero-shot caption generation

**English COCO-style caption**

```text
Describe the content of this image in exactly one fluent English sentence, in the style of an MS COCO image caption. Output only the caption sentence, with no commentary.
```

**Arabic COCO-style caption**

```text
صف محتوى هذه الصورة في جملة عربية فصيحة واحدة بأسلوب وصف الصور الشائع في مجموعة بيانات كوكو. أخرج الجملة العربية فقط دون أي تعليق أو ترجمة.
```

**Arabic concise ablation**

```text
صف محتوى هذه الصورة في جملة عربية فصيحة واحدة قصيرة، لا تزيد عن 12 كلمة. أخرج الوصف فقط دون أي تعليق أو ترجمة.
```

**Arabic grounded ablation**

```text
صف فقط ما هو واضح ومرئي في هذه الصورة في جملة عربية فصيحة واحدة. تجنب التخمين والتفاصيل غير المؤكدة ولا تذكر كائنات أو علاقات لا تستطيع رؤيتها بوضوح. أخرج الوصف فقط.
```

## Translation and cross-lingual consistency

**English VLM output to Arabic**

```text
Translate this English image caption into one fluent Modern Standard Arabic image caption. Preserve the described objects, actions, counts, and relations. Output only the Arabic caption, with no explanation:

{caption}
```

**Arabic to English back-translation**

```text
Back-translate this Arabic image caption into natural English.
Output only the English caption, with no explanation:
{arabic}
```

**English/Arabic VLM consistency judge**

```text
You evaluate whether two English image captions describe the same visual content. Ignore harmless wording, tense, article, and detail differences. Penalise contradictory objects, actions, counts, locations, or relations.

Direct English VLM caption: {english}
Arabic VLM caption after back-translation: {backtranslated}

Return JSON only: {"consistency": <number from 0 to 5>,
"rationale": "<brief explanation>"}.
```

## Text-only LLM-as-judge

This judge receives English references and an Arabic candidate but **no image
pixels**.  In the stored workbook-mode runs, `{image}` is only a generic COCO
image identifier.

```text
You are an expert evaluator of Arabic image caption quality for a research paper. You are given an image, its human reference captions, and one machine-generated caption. Judge only the generated caption.

Image description: {image}
Reference captions:
{references}
Generated caption (to evaluate): {hypot}

Score the generated caption from 0 to 5 (0 = very poor, 5 = excellent) on each dimension:
  fluency:          grammatical correctness and naturalness of the Arabic
  relevance:        how well it matches the content of the references
  object_accuracy:  whether objects are grounded in the references (penalise hallucinations unsupported by the references)
  overall:          your holistic quality score

Return a JSON object ONLY, with keys 'fluency', 'relevance', 'object_accuracy', 'overall'. Each value must be a float between 0 and 5.
```

## Text-only error analysis

```text
You are an Arabic linguistics expert analysing machine-generated image captions for a research paper. Given an image description, reference captions, and one generated caption, identify concrete problems.

Image description: {image}
Reference captions:
{references}
Generated caption: {hypot}

Categorise every problem found. Possible categories:
  hallucination: object/attribute/person/action in the caption that is NOT supported by the references or the image description
  grammar: Arabic morphological or syntactic error (e.g. wrong gender/number agreement, broken construct phrase)
  mt_artifact: clearly machine-translated word order or a literal, unnatural calque of English (translationese)
  repetition: duplicated words or a generic templated phrase
  other: anything else

Return JSON ONLY: {"issues": [{"category": <one of the above>, "span": <quoted word(s) or "caption">, "explanation": <short Arabic/English reason>}], "overall_quality": <0-5 float>, "readability": <0-5 float>}
```

## Image-grounded VLM judges

Both templates below receive the actual COCO image as an attached image input.

**Google-MT caption-resource judge**

```text
You are an expert evaluator of Arabic image captions. You are given an image and one Arabic caption. Judge the caption against what is actually visible in the image.

Arabic caption to evaluate: {caption}

Score from 0 to 5 (0 = very poor, 5 = excellent):
  fluency:          grammatical correctness and naturalness of the Arabic
  relevance:        how well the caption describes the image content
  object_accuracy:  whether every mentioned object/action is really in the image (penalise hallucinations strongly)
  overall:          your holistic quality score

Return JSON ONLY with keys "fluency", "relevance", "object_accuracy", "overall" (floats 0-5) and "hallucinations" (array of the hallucinated Arabic spans, empty array if none).
```

**Prompt-ablation caption judge**

```text
You are an expert evaluator of Arabic image captions. You receive an image and one Arabic caption. Judge the caption against what is actually visible.

Arabic caption: {caption}

Score 0 to 5 (0=very poor, 5=excellent):
- fluency: grammatical correctness and naturalness of Arabic;
- relevance: how well the caption describes the image;
- object_accuracy: whether mentioned objects/actions are visible; penalise hallucinations strongly;
- overall: holistic quality.

Return JSON only with keys fluency, relevance, object_accuracy, overall, and hallucinations (an array of unsupported Arabic spans, or an empty array).
```

## JSON-recovery behavior

For a malformed JSON response, the shared helper retries once at temperature
0 and at most 512 output tokens after appending:

```text
IMPORTANT: Respond with valid JSON ONLY, no prose.
```
