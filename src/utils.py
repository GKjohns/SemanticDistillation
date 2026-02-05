"""
Utility Functions for Semantic Distillation

Caching, logging setup, rate limiting, and other shared utilities.
"""

import json
import sys
import hashlib
import logging
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Any


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str | Path = "logs",
    run_id: str | None = None,
    level: int = logging.INFO,
) -> tuple[logging.Logger, str, Path]:
    """
    Set up logging for a pipeline run.
    
    Args:
        log_dir: Directory to store log files
        run_id: Unique identifier for this run (generated if not provided)
        level: Logging level
    
    Returns:
        Tuple of (logger, run_id, log_file_path)
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = log_dir / f"run_{run_id}.log"
    
    # Create logger
    logger = logging.getLogger("distill")
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    )
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    )
    logger.addHandler(file_handler)
    
    return logger, run_id, log_file


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class FeatureCache:
    """
    Simple file-based cache for extracted features.
    
    Uses MD5 hash of input text as the cache key.
    """
    
    def __init__(self, cache_dir: str | Path = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _cache_key(self, text: str) -> str:
        """Generate a cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _cache_path(self, text: str) -> Path:
        """Get the cache file path for a text."""
        return self.cache_dir / f"{self._cache_key(text)}.json"
    
    def get(self, text: str) -> Optional[dict]:
        """
        Retrieve cached features for the given text.
        
        Returns:
            The cached feature dict, or None if not cached
        """
        path = self._cache_path(text)
        if path.exists():
            return json.loads(path.read_text())
        return None
    
    def set(self, text: str, features: dict) -> None:
        """Cache features for the given text."""
        path = self._cache_path(text)
        path.write_text(json.dumps(features, indent=2))
    
    def has(self, text: str) -> bool:
        """Check if features are cached for the given text."""
        return self._cache_path(text).exists()
    
    def clear(self) -> int:
        """
        Clear all cached features.
        
        Returns:
            Number of cache files removed
        """
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count
    
    def size(self) -> int:
        """Return the number of cached items."""
        return len(list(self.cache_dir.glob("*.json")))


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Thread-safe token bucket rate limiter.
    
    Allows bursting up to `bucket_size` requests, then refills tokens
    at `requests_per_minute` rate. Supports true concurrent requests.
    """
    
    def __init__(self, requests_per_minute: float = 500.0, bucket_size: int = 50):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Token refill rate (default 500)
            bucket_size: Maximum burst size (default 50)
        """
        self.requests_per_minute = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self.bucket_size = bucket_size
        self.tokens = bucket_size  # Start with full bucket
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time (call while holding lock)."""
        now = time.time()
        elapsed = now - self.last_refill_time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.bucket_size, self.tokens + new_tokens)
        self.last_refill_time = now
    
    def acquire(self) -> float:
        """
        Acquire permission to make a request, blocking if necessary.
        
        Returns:
            The time spent waiting (in seconds)
        """
        wait_time = 0.0
        
        while True:
            with self.lock:
                self._refill()
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return wait_time
                
                # Calculate wait time for 1 token
                tokens_needed = 1.0 - self.tokens
                wait_needed = tokens_needed / self.refill_rate
            
            # Sleep outside the lock to allow other threads to proceed
            time.sleep(min(wait_needed, 0.1))  # Sleep in small increments
            wait_time += min(wait_needed, 0.1)
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Result Storage
# ---------------------------------------------------------------------------

def save_results(
    results: dict[str, Any],
    log_dir: str | Path,
    run_id: str,
) -> Path:
    """
    Save pipeline results to a JSON file.
    
    Returns:
        Path to the saved results file
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    results_path = log_dir / f"results_{run_id}.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    
    return results_path


def load_results(path: str | Path) -> dict[str, Any]:
    """Load results from a JSON file."""
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Feature Matrix Helpers
# ---------------------------------------------------------------------------

def save_features_csv(
    features_df,  # pd.DataFrame - avoiding import for lightweight module
    log_dir: str | Path,
    run_id: str,
) -> Path:
    """
    Save feature matrix to CSV.
    
    Returns:
        Path to the saved CSV file
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    features_path = log_dir / f"features_{run_id}.csv"
    features_df.to_csv(features_path)
    
    return features_path
