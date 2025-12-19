#!/usr/bin/env python3
"""
Ciemia CLI - Climate FineWeb Filter

Commands:
    sample-label        Sample candidates and label with GPT
    train-fasttext      Train FastText classifier
    stream-filter-upload Full streaming filter and upload to HF
"""

import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ciemia.log')
    ]
)

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Ciemia - Climate FineWeb Filter Pipeline"""
    pass


@cli.command('sample-label')
@click.option('--num-samples', '-n', default=10000, help='Number of samples to label')
@click.option('--output', '-o', default='data/gpt_labels_10k.jsonl', help='Output JSONL path')
@click.option('--model', '-m', default='gpt-4o-mini', help='OpenAI model to use')
@click.option('--max-chars', default=2000, help='Max chars per sample (truncated)')
@click.option('--dataset-config', default='default', help='FineWeb-Edu config')
@click.option('--resume/--no-resume', default=True, help='Resume from existing labels')
@click.option('--rate-limit', default=0.1, help='Delay between API calls (seconds)')
def sample_label(num_samples, output, model, max_chars, dataset_config, resume, rate_limit):
    """Sample candidates from FineWeb and label with GPT."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        click.echo("Error: OPENAI_API_KEY not found in environment", err=True)
        sys.exit(1)

    from src.streaming_filters import iter_candidates
    from src.gpt_labeler import GPTLabeler

    click.echo(f"Starting sample-label pipeline...")
    click.echo(f"  Samples: {num_samples}")
    click.echo(f"  Output: {output}")
    click.echo(f"  Model: {model}")

    # Initialize labeler
    labeler = GPTLabeler(
        api_key=api_key,
        model=model,
        max_chars=max_chars,
        rate_limit_delay=rate_limit
    )

    # Get candidates (we need more than num_samples since some will be filtered)
    # Request 3x to account for duplicates and failures
    candidates = iter_candidates(
        dataset_config=dataset_config,
        max_samples=num_samples * 3,
        log_interval=5000
    )

    # Label
    stats = labeler.label_batch(
        samples=candidates,
        output_path=output,
        num_samples=num_samples,
        resume=resume
    )

    click.echo(f"\nLabeling complete!")
    click.echo(f"  YES labels: {stats['labeled_yes']}")
    click.echo(f"  NO labels: {stats['labeled_no']}")
    click.echo(f"  Failed: {stats['failed']}")
    click.echo(f"  Duplicates skipped: {stats['skipped_duplicate']}")


@cli.command('build-training')
@click.option('--labels', '-l', default='data/gpt_labels_10k.jsonl', help='Input labels JSONL')
@click.option('--train-output', default='data/fasttext_train.txt', help='Training file output')
@click.option('--valid-output', default='data/fasttext_valid.txt', help='Validation file output')
@click.option('--min-chars', default=50, help='Minimum text length')
@click.option('--valid-ratio', default=0.1, help='Validation set ratio')
@click.option('--seed', default=42, help='Random seed')
def build_training(labels, train_output, valid_output, min_chars, valid_ratio, seed):
    """Build FastText training files from GPT labels."""
    from src.fasttext_trainer import build_training_files

    click.echo(f"Building training files...")
    click.echo(f"  Input: {labels}")
    click.echo(f"  Train output: {train_output}")
    click.echo(f"  Valid output: {valid_output}")

    stats = build_training_files(
        labels_path=labels,
        train_path=train_output,
        valid_path=valid_output,
        min_chars=min_chars,
        valid_ratio=valid_ratio,
        seed=seed
    )

    click.echo(f"\nTraining files built!")
    click.echo(f"  Total loaded: {stats['total_loaded']}")
    click.echo(f"  Climate samples: {stats['climate_count']}")
    click.echo(f"  Other samples: {stats['other_count']}")
    click.echo(f"  Train samples: {stats['train_count']}")
    click.echo(f"  Valid samples: {stats['valid_count']}")
    click.echo(f"  Skipped (too short): {stats['skipped_short']}")


@cli.command('train-fasttext')
@click.option('--train', '-t', default='data/fasttext_train.txt', help='Training file')
@click.option('--valid', '-v', default='data/fasttext_valid.txt', help='Validation file')
@click.option('--output', '-o', default='models/fasttext_climate.bin', help='Model output path')
@click.option('--lr', default=0.5, help='Learning rate')
@click.option('--epoch', default=25, help='Number of epochs')
@click.option('--word-ngrams', default=2, help='Max word n-grams')
@click.option('--dim', default=100, help='Word vector dimension')
@click.option('--threshold', default=0.5, help='Classification threshold for evaluation')
def train_fasttext(train, valid, output, lr, epoch, word_ngrams, dim, threshold):
    """Train FastText binary classifier."""
    from src.fasttext_trainer import train_classifier, evaluate_classifier

    click.echo(f"Training FastText classifier...")
    click.echo(f"  Train file: {train}")
    click.echo(f"  Valid file: {valid}")
    click.echo(f"  Output: {output}")
    click.echo(f"  Params: lr={lr}, epoch={epoch}, wordNgrams={word_ngrams}, dim={dim}")

    # Check files exist
    if not Path(train).exists():
        click.echo(f"Error: Training file not found: {train}", err=True)
        click.echo("Run 'build-training' first.", err=True)
        sys.exit(1)

    # Train
    model = train_classifier(
        train_path=train,
        model_path=output,
        lr=lr,
        epoch=epoch,
        word_ngrams=word_ngrams,
        dim=dim
    )

    # Evaluate
    if Path(valid).exists():
        click.echo(f"\nEvaluating on validation set (threshold={threshold})...")
        metrics = evaluate_classifier(model, valid, threshold)

        click.echo(f"\nEvaluation Results:")
        click.echo(f"  Samples: {metrics['n_samples']}")
        click.echo(f"  Accuracy: {metrics['accuracy']:.4f}")
        click.echo(f"  Climate Precision: {metrics['climate_precision']:.4f}")
        click.echo(f"  Climate Recall: {metrics['climate_recall']:.4f}")
        click.echo(f"  Climate F1: {metrics['climate_f1']:.4f}")
        click.echo(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        click.echo(f"  TP: {cm['tp']}  FP: {cm['fp']}")
        click.echo(f"  FN: {cm['fn']}  TN: {cm['tn']}")

    click.echo(f"\nModel saved to {output}")


@cli.command('stream-filter-upload')
@click.option('--repo-id', '-r', required=True, help='HuggingFace repo ID (e.g., username/fineweb-climate)')
@click.option('--classifier', '-c', default='models/fasttext_climate.bin', help='Classifier model path')
@click.option('--threshold', default=0.5, help='Classifier probability threshold')
@click.option('--buffer-size', default=5000, help='Records per shard')
@click.option('--max-records', default=None, type=int, help='Maximum records to upload')
@click.option('--dataset-config', default='default', help='FineWeb-Edu config')
@click.option('--resume/--no-resume', default=True, help='Resume from previous state')
def stream_filter_upload(repo_id, classifier, threshold, buffer_size, max_records, dataset_config, resume):
    """Stream filter FineWeb and upload to HuggingFace."""
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        click.echo("Error: HF_TOKEN not found in environment", err=True)
        sys.exit(1)

    if not Path(classifier).exists():
        click.echo(f"Error: Classifier model not found: {classifier}", err=True)
        click.echo("Run 'train-fasttext' first.", err=True)
        sys.exit(1)

    from src.streaming_filters import iter_candidates
    from src.uploader import stream_filter_upload as do_upload

    click.echo(f"Starting stream-filter-upload pipeline...")
    click.echo(f"  Repo: {repo_id}")
    click.echo(f"  Classifier: {classifier}")
    click.echo(f"  Threshold: {threshold}")
    click.echo(f"  Buffer size: {buffer_size}")
    click.echo(f"  Max records: {max_records or 'unlimited'}")

    # Get candidates
    candidates = iter_candidates(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config=dataset_config,
        max_samples=None,  # No limit on input
        log_interval=10000
    )

    # Upload
    stats = do_upload(
        samples=candidates,
        repo_id=repo_id,
        hf_token=hf_token,
        classifier_model_path=classifier,
        classifier_threshold=threshold,
        buffer_size=buffer_size,
        max_records=max_records,
        resume=resume
    )

    click.echo(f"\nUpload complete!")
    click.echo(f"  Total processed: {stats['total_processed']}")
    click.echo(f"  Passed classifier: {stats['passed_classifier']}")
    click.echo(f"  Uploaded: {stats['uploaded']}")


@cli.command('download-lid')
def download_lid():
    """Download FastText language identification model."""
    import subprocess

    click.echo("Downloading FastText lid.176.bin model...")

    # Create models directory
    Path("models").mkdir(exist_ok=True)

    # Download using huggingface-cli
    cmd = [
        "huggingface-cli", "download",
        "facebook/fasttext-language-identification",
        "lid.176.bin",
        "--local-dir", "models"
    ]

    try:
        subprocess.run(cmd, check=True)
        click.echo("Download complete! Model saved to models/lid.176.bin")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error downloading model: {e}", err=True)
        sys.exit(1)
    except FileNotFoundError:
        click.echo("Error: huggingface-cli not found. Install with: pip install huggingface_hub", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
