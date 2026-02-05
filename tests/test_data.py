"""
Tests for data.py - Data loading and sample data
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.data import (
    SAMPLE_AITA_POSTS,
    get_sample_aita_data,
    load_data,
)


class TestSampleData:
    """Test sample AITA data"""
    
    def test_sample_posts_structure(self):
        """Test that sample posts have the expected structure"""
        assert len(SAMPLE_AITA_POSTS) > 0
        
        for post in SAMPLE_AITA_POSTS:
            assert "id" in post
            assert "body" in post
            assert "verdict" in post
            assert isinstance(post["id"], str)
            assert isinstance(post["body"], str)
            assert isinstance(post["verdict"], str)
    
    def test_sample_posts_verdicts(self):
        """Test that sample posts have valid verdicts"""
        valid_verdicts = {"NTA", "YTA", "NAH", "ESH"}
        for post in SAMPLE_AITA_POSTS:
            assert post["verdict"] in valid_verdicts
    
    def test_get_sample_aita_data_returns_dataframe(self):
        """Test that get_sample_aita_data returns a DataFrame"""
        df = get_sample_aita_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(SAMPLE_AITA_POSTS)
    
    def test_get_sample_aita_data_columns(self):
        """Test that DataFrame has expected columns"""
        df = get_sample_aita_data()
        assert "id" in df.columns
        assert "body" in df.columns
        assert "verdict" in df.columns


class TestLoadData:
    """Test data loading function"""
    
    def test_load_data_without_path_returns_sample(self):
        """Test that load_data returns sample data when no path provided"""
        df = load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "body" in df.columns
        assert "verdict" in df.columns
    
    def test_load_data_from_csv(self, tmp_path):
        """Test loading data from a CSV file"""
        # Create a test CSV
        test_csv = tmp_path / "test_data.csv"
        test_df = pd.DataFrame({
            "body": ["Post 1", "Post 2"],
            "verdict": ["NTA", "YTA"],
            "extra_col": [1, 2]
        })
        test_df.to_csv(test_csv, index=False)
        
        # Load it
        df = load_data(path=str(test_csv))
        assert len(df) == 2
        assert "body" in df.columns
        assert "verdict" in df.columns
    
    def test_load_data_missing_text_column_raises_error(self, tmp_path):
        """Test that missing text column raises ValueError"""
        test_csv = tmp_path / "bad_data.csv"
        test_df = pd.DataFrame({
            "wrong_column": ["Post 1"],
            "verdict": ["NTA"]
        })
        test_df.to_csv(test_csv, index=False)
        
        with pytest.raises(ValueError, match="Text column 'body' not found"):
            load_data(path=str(test_csv))
    
    def test_load_data_missing_label_column_raises_error(self, tmp_path):
        """Test that missing label column raises ValueError"""
        test_csv = tmp_path / "bad_data.csv"
        test_df = pd.DataFrame({
            "body": ["Post 1"],
            "wrong_label": ["NTA"]
        })
        test_df.to_csv(test_csv, index=False)
        
        with pytest.raises(ValueError, match="Label column 'verdict' not found"):
            load_data(path=str(test_csv))
    
    def test_load_data_binary_labels_conversion(self, tmp_path):
        """Test binary label conversion (NAH→NTA, drop ESH)"""
        test_csv = tmp_path / "test_data.csv"
        test_df = pd.DataFrame({
            "body": ["Post 1", "Post 2", "Post 3", "Post 4"],
            "verdict": ["NTA", "YTA", "NAH", "ESH"]
        })
        test_df.to_csv(test_csv, index=False)
        
        df = load_data(path=str(test_csv), binary_labels=True)
        
        # ESH should be dropped
        assert len(df) == 3
        
        # NAH should be converted to NTA
        assert "NAH" not in df["verdict"].values
        assert "ESH" not in df["verdict"].values
        assert list(df["verdict"]) == ["NTA", "YTA", "NTA"]
    
    def test_load_data_no_binary_conversion(self, tmp_path):
        """Test that binary_labels=False keeps original labels"""
        test_csv = tmp_path / "test_data.csv"
        test_df = pd.DataFrame({
            "body": ["Post 1", "Post 2"],
            "verdict": ["NAH", "ESH"]
        })
        test_df.to_csv(test_csv, index=False)
        
        df = load_data(path=str(test_csv), binary_labels=False)
        
        # Should keep all rows and original labels
        assert len(df) == 2
        assert "NAH" in df["verdict"].values
        assert "ESH" in df["verdict"].values
    
    def test_load_data_custom_column_names(self, tmp_path):
        """Test loading with custom column names"""
        test_csv = tmp_path / "test_data.csv"
        test_df = pd.DataFrame({
            "text_content": ["Post 1"],
            "label": ["NTA"]
        })
        test_df.to_csv(test_csv, index=False)
        
        df = load_data(
            path=str(test_csv),
            text_col="text_content",
            label_col="label",
            binary_labels=False
        )
        assert len(df) == 1
        assert "text_content" in df.columns
        assert "label" in df.columns
