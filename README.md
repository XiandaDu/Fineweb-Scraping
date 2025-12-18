# Ciemia - Climate & Nature Content Filter for FineWeb-Edu

A streaming pipeline to filter climate and nature-related content from the HuggingFace FineWeb-Edu dataset using FastText classifiers and GPT-based weak supervision.

## Overview

This tool:
1. Streams FineWeb-Edu from HuggingFace (no full download required)
2. Filters for English content using FastText language detection
3. Applies keyword filtering for climate/nature topics
4. Uses a trained FastText classifier for high-precision filtering
5. Uploads filtered data to your HuggingFace dataset repository in shards

## Installation

```bash
# Clone or navigate to the project
cd Ciemia

# Install dependencies
pip install -r requirements.txt

# Download the FastText language identification model
python cli.py download-lid
```

## Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_write_token_here
```

- **OPENAI_API_KEY**: Required for GPT labeling (Step 2)
- **HF_TOKEN**: Required for uploading to HuggingFace (Step 5). Must have **write** permissions.

## Usage

### Quick Start (If models are already trained)

If you already have a trained classifier (`models/fasttext_climate.bin`), you can directly run:

```bash
python cli.py stream-filter-upload --repo-id your-username/your-dataset-name
```

### Full Pipeline

#### Step 1: Sample and Label with GPT

Sample candidates from FineWeb-Edu and label them with GPT for weak supervision:

```bash
python cli.py sample-label --num-samples 10000
```

Options:
- `--num-samples, -n`: Number of samples to label (default: 10000)
- `--output, -o`: Output path (default: `data/gpt_labels_10k.jsonl`)
- `--model, -m`: OpenAI model (default: `gpt-4o-mini`)
- `--max-chars`: Max characters per sample (default: 2000)
- `--resume/--no-resume`: Resume from existing labels (default: resume)

#### Step 2: Build Training Files

Convert GPT labels to FastText training format:

```bash
python cli.py build-training
```

Options:
- `--labels, -l`: Input labels file (default: `data/gpt_labels_10k.jsonl`)
- `--train-output`: Training file path (default: `data/fasttext_train.txt`)
- `--valid-output`: Validation file path (default: `data/fasttext_valid.txt`)
- `--min-chars`: Minimum text length (default: 50)
- `--valid-ratio`: Validation split ratio (default: 0.1)

#### Step 3: Train FastText Classifier

Train a binary classifier on the labeled data:

```bash
python cli.py train-fasttext
```

Options:
- `--train, -t`: Training file (default: `data/fasttext_train.txt`)
- `--valid, -v`: Validation file (default: `data/fasttext_valid.txt`)
- `--output, -o`: Model output path (default: `models/fasttext_climate.bin`)
- `--lr`: Learning rate (default: 0.5)
- `--epoch`: Number of epochs (default: 25)
- `--word-ngrams`: Max word n-grams (default: 2)
- `--dim`: Word vector dimension (default: 100)
- `--threshold`: Classification threshold for evaluation (default: 0.5)

#### Step 4: Stream Filter and Upload

Filter the full dataset and upload to HuggingFace:

```bash
python cli.py stream-filter-upload --repo-id your-username/your-dataset-name
```

Options:
- `--repo-id, -r`: HuggingFace dataset repo ID (required)
- `--classifier, -c`: Classifier model path (default: `models/fasttext_climate.bin`)
- `--threshold`: Classification probability threshold (default: 0.5)
- `--buffer-size`: Records per shard before upload (default: 5000)
- `--max-records`: Maximum records to upload (default: unlimited)
- `--resume/--no-resume`: Resume from previous state (default: resume)

### Utility Commands

Download the FastText language ID model:

```bash
python cli.py download-lid
```

## Project Structure

```
Ciemia/
├── cli.py                      # Main CLI entry point
├── requirements.txt            # Python dependencies
├── .env                        # API keys (create this)
├── data/
│   ├── keywords.txt            # Climate/nature keywords for filtering
│   ├── gpt_labels_10k.jsonl    # GPT-labeled samples
│   ├── fasttext_train.txt      # Training data
│   ├── fasttext_valid.txt      # Validation data
│   ├── state.json              # Upload resume state
│   └── temp_shards/            # Temporary shard files
├── models/
│   ├── lid.176.bin             # FastText language ID model
│   └── fasttext_climate.bin    # Trained climate classifier
├── prompts/
│   └── climate_yesno.txt       # GPT labeling prompt
└── src/
    ├── __init__.py
    ├── streaming_filters.py    # English + keyword filtering
    ├── gpt_labeler.py          # GPT labeling logic
    ├── fasttext_trainer.py     # Classifier training + prediction
    └── uploader.py             # HuggingFace upload logic
```

## Pipeline Architecture

```
FineWeb-Edu (streaming)
        │
        ▼
┌───────────────────┐
│  Stage A: English │  FastText lid.176.bin (prob >= 0.9)
│  Language Filter  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Stage B: Keyword │  112 climate/nature keywords
│  Filter           │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Stage C: Climate │  Trained FastText classifier
│  Classifier       │  (prob >= threshold)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Upload to        │  Sharded JSONL.gz files
│  HuggingFace      │
└───────────────────┘
```

## Keywords

The pipeline uses 112 keywords organized into categories:

- **Climate Strong (52)**: climate change, greenhouse gas, IPCC, Paris Agreement, etc.
- **Climate Weak (23)**: adaptation, climate policy, food security, etc.
- **Nature Strong (34)**: biodiversity, ecosystem, conservation, deforestation, etc.
- **Nature Weak (3)**: ecological, ecosystem health, biodiversity loss

Edit `data/keywords.txt` to customize the keyword filter.

## Resume Support

The upload process supports resuming from interruptions:

- State is saved to `data/state.json` after each shard upload
- Run with `--resume` (default) to continue from where you left off
- Run with `--no-resume` to start fresh

## Output Format

Each uploaded record contains:

```json
{
  "text": "The full document text...",
  "id": "Original document ID",
  "url": "Source URL",
  "climate_prob": 0.9234,
  "source": "fineweb"
}
```

## Example Workflow

```bash
# 1. Setup
pip install -r requirements.txt
python cli.py download-lid

# 2. Create .env with your keys
echo "OPENAI_API_KEY=sk-..." > .env
echo "HF_TOKEN=hf_..." >> .env

# 3. Sample and label (takes ~1-2 hours for 10k samples)
python cli.py sample-label

# 4. Build training files
python cli.py build-training

# 5. Train classifier
python cli.py train-fasttext

# 6. Upload to HuggingFace (can run indefinitely)
python cli.py stream-filter-upload --repo-id username/fineweb-climate-filtered

# Or with limits for testing
python cli.py stream-filter-upload --repo-id username/dataset --max-records 1000 --buffer-size 500
```

## License

MIT
