"""
FastText classifier training and prediction.
"""

import json
import logging
import random
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import fasttext
import numpy as np

logger = logging.getLogger(__name__)

# Suppress FastText warnings
fasttext.FastText.eprint = lambda x: None


def safe_predict(model, text: str, k: int = 1):
    """
    Safely call FastText predict with NumPy 2.x compatibility.
    Returns: (labels_tuple, probs_list) where probs are floats
    """
    clean = text.replace("\n", " ").strip()
    if not clean:
        return (("__label__other",) * k, [0.0] * k)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            labels, probs = model.predict(clean, k=k)
            # Convert probs to a regular Python list of floats
            probs_list = [float(p) for p in probs]
            return labels, probs_list
        except (ValueError, TypeError) as e:
            # Fallback for NumPy 2.x: use model.f directly
            try:
                result = model.f.predict(clean, k, 0.0, "")
                labels = tuple(result[0])
                probs_list = [float(p) for p in result[1]]
                return labels, probs_list
            except Exception:
                # Last resort fallback
                return (("__label__other",) * k, [0.0] * k)


def clean_text(text: str) -> str:
    """Clean text for FastText training."""
    # Remove newlines and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove any remaining control characters
    text = ''.join(char for char in text if char.isprintable() or char == ' ')
    return text.strip()


def build_training_files(
    labels_path: str,
    train_path: str,
    valid_path: str,
    min_chars: int = 50,
    valid_ratio: float = 0.1,
    seed: int = 42
) -> dict:
    """
    Convert GPT labels JSONL to FastText training format.

    Args:
        labels_path: Path to gpt_labels_10k.jsonl
        train_path: Output path for training file
        valid_path: Output path for validation file
        min_chars: Minimum text length to include
        valid_ratio: Fraction of data for validation
        seed: Random seed for split

    Returns:
        Dict with statistics
    """
    random.seed(seed)

    labels_path = Path(labels_path)
    train_path = Path(train_path)
    valid_path = Path(valid_path)

    # Create output directories
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and process labels
    samples = []
    stats = {
        'total_loaded': 0,
        'skipped_short': 0,
        'climate_count': 0,
        'other_count': 0,
        'train_count': 0,
        'valid_count': 0
    }

    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                stats['total_loaded'] += 1

                text = record.get('text', '')
                label = record.get('label', '')

                # Clean text
                cleaned = clean_text(text)

                # Skip short texts
                if len(cleaned) < min_chars:
                    stats['skipped_short'] += 1
                    continue

                # Map label to FastText format
                if label == 'YES':
                    ft_label = '__label__climate'
                    stats['climate_count'] += 1
                elif label == 'NO':
                    ft_label = '__label__other'
                    stats['other_count'] += 1
                else:
                    continue

                samples.append(f"{ft_label} {cleaned}")

            except json.JSONDecodeError:
                continue

    # Shuffle and split
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - valid_ratio))

    train_samples = samples[:split_idx]
    valid_samples = samples[split_idx:]

    stats['train_count'] = len(train_samples)
    stats['valid_count'] = len(valid_samples)

    # Write files
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_samples))

    with open(valid_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_samples))

    logger.info(
        f"Built training files - Train: {stats['train_count']}, Valid: {stats['valid_count']}, "
        f"Climate: {stats['climate_count']}, Other: {stats['other_count']}"
    )

    return stats


def train_classifier(
    train_path: str,
    model_path: str,
    lr: float = 0.5,
    epoch: int = 25,
    word_ngrams: int = 2,
    dim: int = 100,
    min_count: int = 1,
    verbose: int = 2
) -> fasttext.FastText._FastText:
    """
    Train FastText supervised classifier.

    Args:
        train_path: Path to training file
        model_path: Path to save model
        lr: Learning rate
        epoch: Number of epochs
        word_ngrams: Max word n-grams
        dim: Word vector dimension
        min_count: Minimum word count
        verbose: Verbosity level

    Returns:
        Trained FastText model
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training FastText classifier (lr={lr}, epoch={epoch}, wordNgrams={word_ngrams})...")

    model = fasttext.train_supervised(
        input=str(train_path),
        lr=lr,
        epoch=epoch,
        wordNgrams=word_ngrams,
        dim=dim,
        minCount=min_count,
        verbose=verbose
    )

    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    return model


def evaluate_classifier(
    model: fasttext.FastText._FastText,
    valid_path: str,
    threshold: float = 0.5
) -> dict:
    """
    Evaluate classifier on validation set.

    Args:
        model: Trained FastText model
        valid_path: Path to validation file
        threshold: Probability threshold for positive class

    Returns:
        Dict with evaluation metrics
    """
    # Standard FastText evaluation
    n_samples, precision, recall = model.test(str(valid_path))

    # Detailed evaluation with confusion matrix
    tp, fp, tn, fn = 0, 0, 0, 0

    with open(valid_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse label and text
            if line.startswith('__label__climate'):
                true_label = 'climate'
                text = line[len('__label__climate'):].strip()
            elif line.startswith('__label__other'):
                true_label = 'other'
                text = line[len('__label__other'):].strip()
            else:
                continue

            # Predict using safe wrapper for NumPy 2.x compatibility
            labels, probs = safe_predict(model, text, k=2)

            # Find climate probability
            climate_prob = 0.0
            for i, label in enumerate(labels):
                if label == '__label__climate' and i < len(probs):
                    climate_prob = probs[i]
                    break

            pred_label = 'climate' if climate_prob >= threshold else 'other'

            # Update confusion matrix
            if true_label == 'climate' and pred_label == 'climate':
                tp += 1
            elif true_label == 'other' and pred_label == 'climate':
                fp += 1
            elif true_label == 'other' and pred_label == 'other':
                tn += 1
            else:
                fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    climate_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    climate_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    climate_f1 = 2 * climate_precision * climate_recall / (climate_precision + climate_recall) if (climate_precision + climate_recall) > 0 else 0

    return {
        'n_samples': n_samples,
        'fasttext_precision': precision,
        'fasttext_recall': recall,
        'threshold': threshold,
        'accuracy': accuracy,
        'confusion_matrix': {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        },
        'climate_precision': climate_precision,
        'climate_recall': climate_recall,
        'climate_f1': climate_f1
    }


class ClimateClassifier:
    """FastText-based climate content classifier."""

    def __init__(self, model_path: str = "models/fasttext_climate.bin", threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Classifier model not found at {self.model_path}")
            self._model = fasttext.load_model(str(self.model_path))
        return self._model

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict climate relevance.

        Returns:
            Tuple of (label, probability)
        """
        cleaned = clean_text(text)
        labels, probs = safe_predict(self.model, cleaned, k=2)

        # Find climate probability
        climate_prob = 0.0
        for i, label in enumerate(labels):
            if label == '__label__climate' and i < len(probs):
                climate_prob = probs[i]
                break

        if climate_prob >= self.threshold:
            return 'climate', climate_prob
        else:
            return 'other', 1 - climate_prob

    def is_climate(self, text: str) -> bool:
        """Check if text is climate-related."""
        label, _ = self.predict(text)
        return label == 'climate'

    def get_climate_prob(self, text: str) -> float:
        """Get probability of text being climate-related."""
        cleaned = clean_text(text)
        labels, probs = safe_predict(self.model, cleaned, k=2)

        for i, label in enumerate(labels):
            if label == '__label__climate' and i < len(probs):
                return probs[i]

        return 0.0
