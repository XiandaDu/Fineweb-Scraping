"""
Streaming filters for FineWeb dataset.
Implements English language detection and keyword filtering.
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple

import fasttext
import numpy as np

logger = logging.getLogger(__name__)

# Suppress FastText warnings
fasttext.FastText.eprint = lambda x: None


def safe_predict(model, text: str, k: int = 1) -> Tuple[tuple, tuple]:
    """
    Safely call FastText predict with NumPy 2.x compatibility.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return model.predict(text, k=k)
        except ValueError:
            # Fallback for NumPy 2.x compatibility
            result = model.f.predict(model.f.getWords(), text, k, 0.0)
            labels = result[0]
            probs = np.asarray(result[1])
            return labels, probs


class LanguageFilter:
    """FastText-based language identification filter."""

    def __init__(self, model_path: str = "models/lid.176.bin", prob_threshold: float = 0.9):
        self.model_path = Path(model_path)
        self.prob_threshold = prob_threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Language ID model not found at {self.model_path}. "
                    "Download with: huggingface-cli download facebook/fasttext-language-identification lid.176.bin --local-dir models/"
                )
            self._model = fasttext.load_model(str(self.model_path))
        return self._model

    def is_english(self, text: str) -> bool:
        """Check if text is English with confidence >= threshold."""
        # Clean text for prediction (single line, no newlines)
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            return False

        try:
            predictions = safe_predict(self.model, clean_text, k=1)
            label, prob = predictions[0][0], predictions[1][0]
        except Exception as e:
            logger.debug(f"Language detection error: {e}")
            return False

        # Label format is __label__eng_Latn (HuggingFace model) or __label__en (original model)
        is_en = label in ("__label__eng_Latn", "__label__en") and prob >= self.prob_threshold
        return is_en


class KeywordFilter:
    """Keyword-based filter for climate/weather content."""

    def __init__(self, keywords_path: str = "data/keywords.txt"):
        self.keywords_path = Path(keywords_path)
        self._keywords: Optional[Set[str]] = None
        self._pattern: Optional[re.Pattern] = None

    @property
    def keywords(self) -> Set[str]:
        if self._keywords is None:
            self._keywords = self._load_keywords()
        return self._keywords

    @property
    def pattern(self) -> re.Pattern:
        if self._pattern is None:
            # Build regex pattern for efficient matching
            escaped = [re.escape(kw) for kw in self.keywords]
            # Use word boundaries for better matching
            pattern_str = r'\b(' + '|'.join(escaped) + r')\b'
            self._pattern = re.compile(pattern_str, re.IGNORECASE)
        return self._pattern

    def _load_keywords(self) -> Set[str]:
        """Load keywords from file."""
        if not self.keywords_path.exists():
            raise FileNotFoundError(f"Keywords file not found at {self.keywords_path}")

        keywords = set()
        with open(self.keywords_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    keywords.add(line.lower())
        return keywords

    def matches(self, text: str) -> bool:
        """Check if text contains any climate/weather keywords."""
        return bool(self.pattern.search(text))


def iter_candidates(
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_config: str = "sample-10BT",
    split: str = "train",
    lang_model_path: str = "models/lid.176.bin",
    keywords_path: str = "data/keywords.txt",
    english_prob_threshold: float = 0.9,
    max_samples: Optional[int] = None,
    log_interval: int = 10000
) -> Iterator[dict]:
    """
    Stream FineWeb and yield samples passing English + keyword filters.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (e.g., 'sample-10BT')
        split: Dataset split
        lang_model_path: Path to FastText language ID model
        keywords_path: Path to keywords file
        english_prob_threshold: Minimum probability for English detection
        max_samples: Maximum samples to process (None for unlimited)
        log_interval: How often to log progress

    Yields:
        Dict with 'text' and metadata for each matching sample
    """
    from datasets import load_dataset

    logger.info(f"Loading dataset {dataset_name}/{dataset_config} in streaming mode...")

    # Load dataset in streaming mode
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True
    )

    # Initialize filters
    lang_filter = LanguageFilter(lang_model_path, english_prob_threshold)
    keyword_filter = KeywordFilter(keywords_path)

    # Statistics
    total_processed = 0
    passed_english = 0
    passed_keywords = 0

    logger.info("Starting streaming filter...")

    for sample in dataset:
        total_processed += 1

        text = sample.get('text', '')
        if not text:
            continue

        # Stage A: English filter
        if not lang_filter.is_english(text):
            continue
        passed_english += 1

        # Stage B: Keyword filter
        if not keyword_filter.matches(text):
            continue
        passed_keywords += 1

        # Yield matching sample with metadata
        yield {
            'text': text,
            'id': sample.get('id', f'sample_{total_processed}'),
            'url': sample.get('url', ''),
            'dump': sample.get('dump', ''),
            'date': sample.get('date', ''),
        }

        # Log progress
        if total_processed % log_interval == 0:
            eng_rate = passed_english / total_processed * 100
            kw_rate = passed_keywords / total_processed * 100
            logger.info(
                f"Processed: {total_processed:,} | "
                f"English: {passed_english:,} ({eng_rate:.2f}%) | "
                f"Keywords: {passed_keywords:,} ({kw_rate:.2f}%)"
            )

        # Check limit
        if max_samples and passed_keywords >= max_samples:
            logger.info(f"Reached max_samples limit: {max_samples}")
            break

    # Final statistics
    if total_processed > 0:
        eng_rate = passed_english / total_processed * 100
        kw_rate = passed_keywords / total_processed * 100
        logger.info(
            f"Final stats - Processed: {total_processed:,} | "
            f"English: {passed_english:,} ({eng_rate:.2f}%) | "
            f"Keywords: {passed_keywords:,} ({kw_rate:.2f}%)"
        )
