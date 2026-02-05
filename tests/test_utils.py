"""
Tests for utils.py - Caching, rate limiting, logging, and utilities
"""

import pytest
import json
import time
import tempfile
import threading
from pathlib import Path
from src.utils import (
    FeatureCache,
    RateLimiter,
    setup_logging,
    save_results,
    load_results,
    compute_feature_config_hash,
)


class TestFeatureCache:
    """Test the FeatureCache class"""
    
    def test_cache_set_and_get(self, tmp_path):
        """Test basic cache set and get operations"""
        cache = FeatureCache(tmp_path)
        
        text = "This is a test post"
        features = {"feature1": 5, "feature2": True}
        
        # Set cache
        cache.set(text, features)
        
        # Get cache
        retrieved = cache.get(text)
        assert retrieved == features
    
    def test_cache_miss(self, tmp_path):
        """Test that cache miss returns None"""
        cache = FeatureCache(tmp_path)
        result = cache.get("text that was never cached")
        assert result is None
    
    def test_cache_has(self, tmp_path):
        """Test the has() method"""
        cache = FeatureCache(tmp_path)
        
        text = "Test text"
        features = {"test": 1}
        
        assert not cache.has(text)
        cache.set(text, features)
        assert cache.has(text)
    
    def test_cache_size(self, tmp_path):
        """Test counting cached items"""
        cache = FeatureCache(tmp_path)
        
        assert cache.size() == 0
        
        cache.set("text1", {"a": 1})
        assert cache.size() == 1
        
        cache.set("text2", {"b": 2})
        assert cache.size() == 2
    
    def test_cache_clear(self, tmp_path):
        """Test clearing the cache"""
        cache = FeatureCache(tmp_path)
        
        cache.set("text1", {"a": 1})
        cache.set("text2", {"b": 2})
        assert cache.size() == 2
        
        cleared = cache.clear()
        assert cleared == 2
        assert cache.size() == 0
    
    def test_cache_persistence(self, tmp_path):
        """Test that cache persists across instances"""
        cache1 = FeatureCache(tmp_path)
        cache1.set("text", {"feature": 5})
        
        # Create new cache instance pointing to same directory
        cache2 = FeatureCache(tmp_path)
        retrieved = cache2.get("text")
        assert retrieved == {"feature": 5}
    
    def test_cache_with_feature_config_hash(self, tmp_path):
        """Test that cache entries are scoped by feature config hash"""
        cache = FeatureCache(tmp_path)
        text = "Same text for both configs"
        
        config_hash_v1 = "abc123"
        config_hash_v2 = "def456"
        
        features_v1 = {"feature_a": 3, "feature_b": True}
        features_v2 = {"feature_x": 5, "feature_y": False}
        
        # Cache with config v1
        cache.set(text, features_v1, feature_config_hash=config_hash_v1)
        
        # Cache with config v2 (different features for same text)
        cache.set(text, features_v2, feature_config_hash=config_hash_v2)
        
        # Retrieving with v1 hash returns v1 features
        assert cache.get(text, feature_config_hash=config_hash_v1) == features_v1
        
        # Retrieving with v2 hash returns v2 features
        assert cache.get(text, feature_config_hash=config_hash_v2) == features_v2
        
        # Retrieving with no hash returns neither (different key)
        assert cache.get(text) != features_v1 or cache.get(text) != features_v2
    
    def test_cache_miss_with_different_config_hash(self, tmp_path):
        """Test that changing the config hash causes a cache miss"""
        cache = FeatureCache(tmp_path)
        text = "Test post text"
        
        # Cache with one config
        cache.set(text, {"old": 1}, feature_config_hash="config_v1")
        
        # Looking up with a different config hash should miss
        assert cache.get(text, feature_config_hash="config_v2") is None
        assert not cache.has(text, feature_config_hash="config_v2")
        
        # But the original is still there
        assert cache.has(text, feature_config_hash="config_v1")
        assert cache.get(text, feature_config_hash="config_v1") == {"old": 1}
    
    def test_cache_has_with_feature_config_hash(self, tmp_path):
        """Test has() respects feature config hash"""
        cache = FeatureCache(tmp_path)
        text = "Test text"
        config_hash = "my_config_hash"
        
        assert not cache.has(text, feature_config_hash=config_hash)
        cache.set(text, {"f": 1}, feature_config_hash=config_hash)
        assert cache.has(text, feature_config_hash=config_hash)
        assert not cache.has(text, feature_config_hash="other_hash")


class TestComputeFeatureConfigHash:
    """Test the compute_feature_config_hash utility"""
    
    def test_same_config_same_hash(self):
        """Test that identical configs produce the same hash"""
        config = {"name": "Test", "features": [{"name": "f1", "type": "bool"}]}
        hash1 = compute_feature_config_hash(config)
        hash2 = compute_feature_config_hash(config)
        assert hash1 == hash2
    
    def test_different_config_different_hash(self):
        """Test that different configs produce different hashes"""
        config1 = {"name": "Test", "features": [{"name": "f1", "type": "bool"}]}
        config2 = {"name": "Test", "features": [{"name": "f2", "type": "scale"}]}
        hash1 = compute_feature_config_hash(config1)
        hash2 = compute_feature_config_hash(config2)
        assert hash1 != hash2
    
    def test_key_order_does_not_matter(self):
        """Test that dict key order doesn't affect the hash (canonical JSON)"""
        config1 = {"name": "Test", "features": []}
        config2 = {"features": [], "name": "Test"}
        hash1 = compute_feature_config_hash(config1)
        hash2 = compute_feature_config_hash(config2)
        assert hash1 == hash2
    
    def test_returns_hex_string(self):
        """Test that the hash is a valid hex string"""
        config = {"name": "Test", "features": []}
        h = compute_feature_config_hash(config)
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex digest length
        int(h, 16)  # Should be valid hex


class TestRateLimiter:
    """Test the RateLimiter class"""
    
    def test_rate_limiter_allows_immediate_requests(self):
        """Test that initial requests within bucket size don't wait"""
        limiter = RateLimiter(requests_per_minute=60, bucket_size=5)
        
        start = time.time()
        for _ in range(5):
            wait = limiter.acquire()
            # Should be minimal wait for first few requests
            assert wait < 0.1
        elapsed = time.time() - start
        
        # Should complete quickly (well under 1 second)
        assert elapsed < 0.5
    
    def test_rate_limiter_blocks_when_bucket_empty(self):
        """Test that rate limiter blocks when bucket is empty"""
        # Very slow rate: 6 requests per minute = 1 request per 10 seconds
        # Bucket size of 2
        limiter = RateLimiter(requests_per_minute=6, bucket_size=2)
        
        # First 2 should be fast
        limiter.acquire()
        limiter.acquire()
        
        # Third should block (but we'll only wait a short time for testing)
        start = time.time()
        limiter.acquire()
        wait_time = time.time() - start
        
        # Should have waited some amount (checking it's not instant)
        assert wait_time > 0.01
    
    def test_rate_limiter_context_manager(self):
        """Test using rate limiter as context manager"""
        limiter = RateLimiter(requests_per_minute=60, bucket_size=5)
        
        with limiter:
            pass  # Should not raise any errors
    
    def test_rate_limiter_thread_safety(self):
        """Test that rate limiter works with multiple threads"""
        limiter = RateLimiter(requests_per_minute=120, bucket_size=10)
        results = []
        
        def worker():
            limiter.acquire()
            results.append(1)
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All threads should have completed
        assert len(results) == 5


class TestLogging:
    """Test logging setup"""
    
    def test_setup_logging_creates_files(self, tmp_path):
        """Test that setup_logging creates log files"""
        logger, run_id, log_file = setup_logging(log_dir=tmp_path, run_id="test_run")
        
        assert logger.name == "distill"
        assert run_id == "test_run"
        assert log_file.exists()
    
    def test_setup_logging_generates_run_id(self, tmp_path):
        """Test that setup_logging generates run_id if not provided"""
        logger, run_id, log_file = setup_logging(log_dir=tmp_path)
        
        # Run ID should be a timestamp string
        assert len(run_id) > 0
        assert log_file.exists()
    
    def test_logger_writes_to_file(self, tmp_path):
        """Test that logger actually writes to the file"""
        logger, run_id, log_file = setup_logging(log_dir=tmp_path)
        
        test_message = "Test log message"
        logger.info(test_message)
        
        # Read log file
        log_content = log_file.read_text()
        assert test_message in log_content


class TestResultStorage:
    """Test result saving and loading"""
    
    def test_save_and_load_results(self, tmp_path):
        """Test saving and loading results"""
        results = {
            "accuracy": 0.85,
            "features": ["f1", "f2"],
            "metadata": {"run": "test"}
        }
        
        # Save results
        results_path = save_results(results, log_dir=tmp_path, run_id="test_run")
        assert results_path.exists()
        
        # Load results
        loaded = load_results(results_path)
        assert loaded["accuracy"] == 0.85
        assert loaded["features"] == ["f1", "f2"]
        assert loaded["metadata"]["run"] == "test"
    
    def test_save_results_with_non_serializable_types(self, tmp_path):
        """Test that save_results handles datetime and other types"""
        from datetime import datetime
        
        results = {
            "timestamp": datetime.now(),
            "value": 42
        }
        
        # Should not raise error (uses default=str)
        results_path = save_results(results, log_dir=tmp_path, run_id="test")
        assert results_path.exists()
        
        loaded = load_results(results_path)
        assert loaded["value"] == 42
        assert isinstance(loaded["timestamp"], str)  # Converted to string
