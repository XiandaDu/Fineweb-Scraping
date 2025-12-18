"""
GPT-based labeling for climate content classification.
Uses OpenAI API to label samples as YES/NO for climate-related content.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"


def load_prompt(prompt_path: str = "prompts/climate_yesno.txt") -> str:
    """Load the labeling prompt template."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """
    Truncate text to max_chars, keeping head and tail.
    This preserves context from both beginning and end of the document.
    """
    if len(text) <= max_chars:
        return text

    # Keep 70% from head, 30% from tail
    head_chars = int(max_chars * 0.7)
    tail_chars = max_chars - head_chars - 20  # 20 chars for separator

    head = text[:head_chars]
    tail = text[-tail_chars:]

    return f"{head}\n\n[...truncated...]\n\n{tail}"


def text_hash(text: str) -> str:
    """Generate a hash for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def parse_response(response: str) -> Optional[str]:
    """Parse GPT response to extract YES/NO label."""
    response = response.strip().upper()

    if response in ('YES', 'NO'):
        return response

    # Try to extract from longer responses
    if response.startswith('YES'):
        return 'YES'
    if response.startswith('NO'):
        return 'NO'

    return None


class GPTLabeler:
    """Label samples using GPT API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.1,
        max_chars: int = 2000,
        prompt_path: str = "prompts/climate_yesno.txt"
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.max_chars = max_chars
        self.prompt_template = load_prompt(prompt_path)

    def label_single(self, text: str) -> Optional[str]:
        """Label a single text sample."""
        truncated = truncate_text(text, self.max_chars)
        prompt = self.prompt_template.format(text=truncated)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10,
                    temperature=0
                )

                answer = response.choices[0].message.content
                label = parse_response(answer)

                if label is None:
                    logger.warning(f"Could not parse response: {answer}")
                    continue

                return label

            except Exception as e:
                logger.warning(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        return None

    def label_batch(
        self,
        samples: Iterator[dict],
        output_path: str,
        num_samples: int = 10000,
        resume: bool = True
    ) -> dict:
        """
        Label a batch of samples and write to JSONL file.

        Args:
            samples: Iterator yielding dicts with 'text' and 'id'
            output_path: Path to output JSONL file
            num_samples: Number of samples to label
            resume: Whether to resume from existing file

        Returns:
            Dict with statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing labels for resume and deduplication
        existing_hashes = set()
        existing_count = 0

        if resume and output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        existing_hashes.add(record.get('text_hash', ''))
                        existing_count += 1
                    except json.JSONDecodeError:
                        continue

            logger.info(f"Resuming from {existing_count} existing labels")

        # Statistics
        stats = {
            'total_processed': 0,
            'labeled_yes': 0,
            'labeled_no': 0,
            'skipped_duplicate': 0,
            'failed': 0
        }

        # Open file in append mode
        mode = 'a' if resume and existing_count > 0 else 'w'
        labels_needed = num_samples - existing_count

        if labels_needed <= 0:
            logger.info(f"Already have {existing_count} labels, target is {num_samples}")
            return stats

        logger.info(f"Need to label {labels_needed} more samples")

        with open(output_path, mode, encoding='utf-8') as f:
            pbar = tqdm(total=labels_needed, desc="Labeling")

            for sample in samples:
                if stats['labeled_yes'] + stats['labeled_no'] >= labels_needed:
                    break

                text = sample.get('text', '')
                sample_id = sample.get('id', '')

                # Check for duplicates
                h = text_hash(text)
                if h in existing_hashes:
                    stats['skipped_duplicate'] += 1
                    continue

                existing_hashes.add(h)
                stats['total_processed'] += 1

                # Label with GPT
                label = self.label_single(text)

                if label is None:
                    stats['failed'] += 1
                    continue

                # Update stats
                if label == 'YES':
                    stats['labeled_yes'] += 1
                else:
                    stats['labeled_no'] += 1

                # Write record
                record = {
                    'id': sample_id,
                    'text': text,
                    'text_hash': h,
                    'label': label,
                    'model': self.model,
                    'prompt_version': PROMPT_VERSION,
                    'timestamp': datetime.utcnow().isoformat()
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()

                pbar.update(1)

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            pbar.close()

        logger.info(
            f"Labeling complete - YES: {stats['labeled_yes']}, NO: {stats['labeled_no']}, "
            f"Failed: {stats['failed']}, Duplicates: {stats['skipped_duplicate']}"
        )

        return stats
