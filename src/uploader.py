"""
Streaming upload to Hugging Face Hub.
Handles buffering, shard creation, upload, and resume functionality.
"""

import gzip
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from huggingface_hub import HfApi, create_repo

logger = logging.getLogger(__name__)


class StreamingUploader:
    """
    Uploads filtered data to Hugging Face in shards.
    Maintains minimal local storage by uploading and deleting shards.
    """

    def __init__(
        self,
        repo_id: str,
        token: str,
        buffer_size: int = 5000,
        state_path: str = "data/state.json",
        temp_dir: str = "data/temp_shards"
    ):
        self.repo_id = repo_id
        self.token = token
        self.buffer_size = buffer_size
        self.state_path = Path(state_path)
        self.temp_dir = Path(temp_dir)
        self.api = HfApi(token=token)

        # State
        self.buffer = []
        self.shard_index = 0
        self.total_uploaded = 0
        self.start_time = None

        # Create directories
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> dict:
        """Load resume state from file."""
        if self.state_path.exists():
            with open(self.state_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        """Save current state for resume."""
        state = {
            'shard_index': self.shard_index,
            'total_uploaded': self.total_uploaded,
            'last_update': datetime.utcnow().isoformat()
        }
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def _create_repo_if_needed(self):
        """Create the HuggingFace repo if it doesn't exist."""
        try:
            create_repo(
                repo_id=self.repo_id,
                token=self.token,
                repo_type="dataset",
                exist_ok=True
            )
            logger.info(f"Repository {self.repo_id} ready")
        except Exception as e:
            logger.warning(f"Could not create repo (may already exist): {e}")

    def _write_shard(self, records: list) -> Path:
        """Write records to a temporary shard file."""
        shard_name = f"shard_{self.shard_index:05d}.jsonl.gz"
        shard_path = self.temp_dir / shard_name

        with gzip.open(shard_path, 'wt', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        return shard_path

    def _upload_shard(self, shard_path: Path, max_retries: int = 5) -> bool:
        """Upload a shard to HuggingFace and delete local file with retry logic."""
        path_in_repo = f"data/{shard_path.name}"

        for attempt in range(max_retries):
            try:
                self.api.upload_file(
                    path_or_fileobj=str(shard_path),
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="dataset"
                )

                logger.info(f"Uploaded {shard_path.name} ({len(self.buffer)} records)")

                # Delete local file
                shard_path.unlink()

                return True

            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                logger.warning(f"Upload attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to upload {shard_path.name} after {max_retries} attempts")
                    return False

        return False

    def add_record(self, record: dict) -> bool:
        """
        Add a record to the buffer.
        Returns True if a shard was uploaded.
        """
        self.buffer.append(record)

        if len(self.buffer) >= self.buffer_size:
            return self.flush()

        return False

    def flush(self) -> bool:
        """Flush current buffer to a shard and upload."""
        if not self.buffer:
            return False

        # Write shard
        shard_path = self._write_shard(self.buffer)

        # Upload
        success = self._upload_shard(shard_path)

        if success:
            self.total_uploaded += len(self.buffer)
            self.shard_index += 1
            self.buffer = []
            self._save_state()

        return success

    def resume(self) -> dict:
        """
        Resume from previous state.
        Returns state dict with shard_index and total_uploaded.
        """
        state = self._load_state()

        if state:
            self.shard_index = state.get('shard_index', 0)
            self.total_uploaded = state.get('total_uploaded', 0)
            logger.info(
                f"Resuming from shard {self.shard_index}, "
                f"{self.total_uploaded} records previously uploaded"
            )

        return state

    def get_stats(self) -> dict:
        """Get upload statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            'shards_uploaded': self.shard_index,
            'total_records': self.total_uploaded,
            'buffer_size': len(self.buffer),
            'elapsed_seconds': elapsed,
            'records_per_second': self.total_uploaded / elapsed if elapsed > 0 else 0
        }


def stream_filter_upload(
    samples: Iterator[dict],
    repo_id: str,
    hf_token: str,
    classifier_model_path: str = "models/fasttext_climate.bin",
    classifier_threshold: float = 0.5,
    buffer_size: int = 5000,
    max_records: Optional[int] = None,
    resume: bool = True,
    log_interval: int = 1000
) -> dict:
    """
    Stream samples through classifier and upload to HuggingFace.

    Args:
        samples: Iterator from iter_candidates()
        repo_id: HuggingFace dataset repo ID
        hf_token: HuggingFace API token
        classifier_model_path: Path to trained FastText classifier
        classifier_threshold: Probability threshold for climate classification
        buffer_size: Records per shard
        max_records: Maximum records to upload (None for unlimited)
        resume: Whether to resume from previous state
        log_interval: How often to log progress

    Returns:
        Dict with statistics
    """
    from .fasttext_trainer import ClimateClassifier

    # Initialize classifier
    classifier = ClimateClassifier(classifier_model_path, classifier_threshold)

    # Initialize uploader
    uploader = StreamingUploader(
        repo_id=repo_id,
        token=hf_token,
        buffer_size=buffer_size
    )

    # Create repo
    uploader._create_repo_if_needed()

    # Resume if needed
    resume_state = {}
    if resume:
        resume_state = uploader.resume()
        # Note: We can't reliably skip samples in the iterator since filtering
        # means processed count != uploaded count. Resume is handled at shard level.

    uploader.start_time = time.time()

    # Statistics
    stats = {
        'total_processed': 0,
        'passed_classifier': 0,
        'uploaded': 0
    }

    logger.info(f"Starting stream-filter-upload to {repo_id}...")
    logger.info("Waiting for first sample from iterator...")

    try:
        sample_count = 0
        for sample in samples:
            sample_count += 1
            if sample_count == 1:
                logger.info("Received first sample from iterator! Starting processing...")
            stats['total_processed'] += 1

            try:
                text = sample.get('text', '')
                if not text:
                    continue

                # Stage C: Apply trained classifier
                if not classifier.is_climate(text):
                    continue

                stats['passed_classifier'] += 1

                # Get probability for metadata
                climate_prob = classifier.get_climate_prob(text)

                # Build record for upload
                record = {
                    'text': text,
                    'id': sample.get('id', ''),
                    'url': sample.get('url', ''),
                    'climate_prob': round(climate_prob, 4),
                    'source': 'fineweb'
                }

                # Add to uploader
                uploaded = uploader.add_record(record)

                if uploaded:
                    stats['uploaded'] = uploader.total_uploaded

                # Check limit
                if max_records and stats['passed_classifier'] >= max_records:
                    logger.info(f"Reached max_records limit: {max_records}")
                    break

                # Log progress
                if stats['total_processed'] % log_interval == 0:
                    pass_rate = stats['passed_classifier'] / stats['total_processed'] * 100 if stats['total_processed'] > 0 else 0
                    upload_stats = uploader.get_stats()
                    logger.info(
                        f"Processed: {stats['total_processed']:,} | "
                        f"Passed: {stats['passed_classifier']:,} ({pass_rate:.2f}%) | "
                        f"Uploaded: {upload_stats['total_records']:,} | "
                        f"Shards: {upload_stats['shards_uploaded']}"
                    )

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                raise
            except Exception as e:
                logger.warning(f"Error processing sample {stats['total_processed']}: {e}")
                continue

    except KeyboardInterrupt:
        logger.info("Interrupted by user - flushing buffer...")
        uploader.flush()
        raise
    except Exception as e:
        logger.error(f"Fatal error in stream processing: {e}", exc_info=True)
        uploader.flush()
        raise

    # Final flush
    uploader.flush()
    stats['uploaded'] = uploader.total_uploaded

    # Final stats
    upload_stats = uploader.get_stats()
    logger.info(
        f"Upload complete - Total processed: {stats['total_processed']:,}, "
        f"Passed classifier: {stats['passed_classifier']:,}, "
        f"Uploaded: {stats['uploaded']:,}, "
        f"Shards: {upload_stats['shards_uploaded']}, "
        f"Time: {upload_stats['elapsed_seconds']:.1f}s"
    )

    return stats
