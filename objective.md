## Goal

Stream FineWeb from Hugging Face without storing the full dataset locally, filter for English, climate / disruptive weather–related text, build a weakly supervised training set using GPT (10k Yes/No labels), train a FastText binary classifier, and stream-filter + upload the final corpus to a Hugging Face dataset repository in shards.

## Pipeline

1. Streaming ingest: Read FineWeb with datasets.load_dataset(..., streaming=True) (no full local storage).

2. Stage A (cheap): FastText language ID (lid.176.bin) → keep confident English only.

3. Stage B (high recall): Keyword / pattern filter for climate & disruptive weather.

4. Weak supervision: Sample 10,000 candidates from A+B and label with GPT (YES/NO climate-related).

5. Train classifier: Train a FastText binary classifier (climate vs other) on GPT-labeled data.

6. Stage C (precision): Re-run streaming A+B, then apply the FastText classifier with a probability threshold.

7. Streaming write & upload: Accumulate small buffers, write temporary shards, and push to Hugging Face; delete shards after upload; support resume.

8. Reproducibility: Log thresholds, keyword list, prompts, model hashes, and per-stage statistics.

## Outputs

data/keywords.txt

data/gpt_labels_10k.jsonl

data/fasttext_train.txt, data/fasttext_valid.txt

models/fasttext_climate.bin

HF dataset repo: <user>/fineweb-climate-filtered


## Procedure
Step 0 — Project skeleton & config

Prompt

You are a senior ML engineer. Create a minimal, runnable Python project skeleton for streaming FineWeb filtering and HF upload.
Requirements: Python, datasets, fasttext, huggingface_hub, dotenv for HF_TOKEN and OPENAI_API_KEY.
Provide a CLI with commands:

sample-label (sample + GPT labeling)

train-fasttext (train classifier)

stream-filter-upload (full streaming filter + upload)
Output: directory tree, file responsibilities, and example commands.

Step 1 — Streaming ingest + English + keyword filter (merged)

Prompt

Implement streaming_filters.py:

Stream FineWeb with streaming=True.

Load FastText lid.176.bin and implement is_english(text, prob>=0.9).

Implement keyword_filter(text, keywords) (case-insensitive).

Expose iter_candidates() yielding samples passing English + keyword filters (include text and any available metadata).

Add logging for totals and pass rates.
Constraint: no full dataset written to disk.

Step 2 — Sample 10k + GPT Yes/No labeling

Prompt

Implement sample-label:

Sample 10,000 items from iter_candidates() (configurable).

Call the OpenAI API to label each item as climate-related (YES/NO only).

Enforce strict output parsing, retries, rate limiting, and optional concurrency.

Truncate text safely (e.g., max 2,000 chars; keep head+tail).

De-duplicate by text hash.

Write data/gpt_labels_10k.jsonl with fields: id, text, label, model, prompt_version, timestamp.

Save the exact prompt to prompts/climate_yesno.txt.

Labeling Prompt Template

You are labeling web text for a dataset.
Question: Is this text substantially about climate, climate change, global warming, extreme or disruptive weather events, or their impacts (e.g., floods, hurricanes, droughts, heatwaves, wildfires, climate risk, adaptation/mitigation)?
Answer ONLY with YES or NO.
Text: {text}

Step 3 — Build FastText training files

Prompt

Convert data/gpt_labels_10k.jsonl to FastText supervised format:

__label__climate <text> or __label__other <text>.

Clean text (remove newlines, normalize whitespace).

Drop very short texts (e.g., <50 chars).

Split into train/valid (e.g., 90/10).
Output data/fasttext_train.txt and data/fasttext_valid.txt.

Step 4 — Train FastText classifier

Prompt

Implement train-fasttext:

Train a binary FastText supervised classifier.

Reasonable defaults (e.g., lr, epoch, wordNgrams), configurable via CLI.

Save models/fasttext_climate.bin.

Evaluate on validation set and print metrics (accuracy, confusion matrix at a configurable threshold).

Step 5 — Full streaming filter + HF upload (no disk accumulation)

Prompt

Implement stream-filter-upload:

Stream FineWeb → English filter → keyword filter → FastText climate classifier (prob >= threshold).

Accumulate a small buffer (e.g., 5k rows), write a temporary shard (parquet or jsonl.gz), immediately upload to a Hugging Face dataset repo using huggingface_hub, then delete the shard.

Support resume with state.json (processed count / shard index).

Log per-stage pass counts, retention rate, throughput, and uploaded shard count.
Constraint: only temporary shard files allowed locally.

Notes for the GPT-labeled weak supervision

Sample after English + keyword filtering to reduce cost and noise.

Keep the prompt fixed and outputs strictly constrained to YES/NO.

Optionally add a small set of hard negatives sampled from English text that fails keywords to improve rejection.