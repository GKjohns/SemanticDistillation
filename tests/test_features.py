"""
Tests for features.py - Feature definitions and FeatureSet class
"""

import pytest
import tempfile
from pathlib import Path
from src.features import (
    DEFAULT_FEATURES,
    FULL_FEATURES,
    MINIMAL_FEATURES,
    FeatureSet,
    create_feature,
)


class TestFeatureDefinitions:
    """Test predefined feature sets"""
    
    def test_default_features_structure(self):
        """Test DEFAULT_FEATURES has expected structure"""
        assert "name" in DEFAULT_FEATURES
        assert "description" in DEFAULT_FEATURES
        assert "features" in DEFAULT_FEATURES
        assert DEFAULT_FEATURES["name"] == "AITAFeatures"
        assert len(DEFAULT_FEATURES["features"]) > 0
    
    def test_all_features_have_required_fields(self):
        """Test that all features have required fields"""
        for feature_set in [DEFAULT_FEATURES, FULL_FEATURES, MINIMAL_FEATURES]:
            for feature in feature_set["features"]:
                assert "name" in feature
                assert "type" in feature
                assert "description" in feature
                assert feature["type"] in ["scale", "bool", "categorical"]
    
    def test_scale_features_have_min_max(self):
        """Test that scale features have min/max values"""
        for feature_set in [DEFAULT_FEATURES, FULL_FEATURES, MINIMAL_FEATURES]:
            for feature in feature_set["features"]:
                if feature["type"] == "scale":
                    assert "min" in feature
                    assert "max" in feature
                    assert feature["min"] < feature["max"]
    
    def test_categorical_features_have_values(self):
        """Test that categorical features have values list"""
        for feature_set in [DEFAULT_FEATURES, FULL_FEATURES, MINIMAL_FEATURES]:
            for feature in feature_set["features"]:
                if feature["type"] == "categorical":
                    assert "values" in feature
                    assert len(feature["values"]) > 0
    
    def test_feature_names_are_unique(self):
        """Test that feature names are unique within each set"""
        for feature_set in [DEFAULT_FEATURES, FULL_FEATURES, MINIMAL_FEATURES]:
            names = [f["name"] for f in feature_set["features"]]
            assert len(names) == len(set(names))


class TestFeatureSet:
    """Test FeatureSet wrapper class"""
    
    def test_featureset_creation(self):
        """Test creating a FeatureSet"""
        fs = FeatureSet(DEFAULT_FEATURES)
        assert fs.name == "AITAFeatures"
        assert len(fs.features) > 0
    
    def test_featureset_properties(self):
        """Test FeatureSet properties"""
        fs = FeatureSet(MINIMAL_FEATURES)
        assert fs.name == "MinimalFeatures"
        assert len(fs.description) > 0
        assert len(fs.features) == 3
        assert "sentiment" in fs.feature_names
        assert "self_aware" in fs.feature_names
        assert "conflict_type" in fs.feature_names
    
    def test_add_feature(self):
        """Test adding a feature to a FeatureSet"""
        # Use a copy to avoid modifying the original
        fs = FeatureSet(MINIMAL_FEATURES).copy()
        initial_count = len(fs.features)
        
        new_feature = {
            "name": "new_test_feature",
            "type": "bool",
            "description": "A new test feature"
        }
        fs.add_feature(new_feature)
        
        assert len(fs.features) == initial_count + 1
        assert "new_test_feature" in fs.feature_names
    
    def test_add_duplicate_feature_raises_error(self):
        """Test that adding duplicate feature raises ValueError"""
        # Use a copy to avoid modifying the original
        fs = FeatureSet(MINIMAL_FEATURES).copy()
        
        duplicate = {
            "name": "sentiment",  # Already exists
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "Duplicate"
        }
        
        with pytest.raises(ValueError, match="already exists"):
            fs.add_feature(duplicate)
    
    def test_remove_feature(self):
        """Test removing a feature from a FeatureSet"""
        # Use a copy to avoid modifying the original
        fs = FeatureSet(MINIMAL_FEATURES).copy()
        initial_count = len(fs.features)
        
        fs.remove_feature("sentiment")
        
        assert len(fs.features) == initial_count - 1
        assert "sentiment" not in fs.feature_names
    
    def test_get_feature(self):
        """Test getting a feature by name"""
        fs = FeatureSet(MINIMAL_FEATURES)
        
        feature = fs.get_feature("sentiment")
        assert feature is not None
        assert feature["name"] == "sentiment"
        assert feature["type"] == "scale"
        
        # Non-existent feature
        assert fs.get_feature("nonexistent") is None
    
    def test_copy(self):
        """Test copying a FeatureSet"""
        # Use a fresh copy from MINIMAL_FEATURES
        fs1 = FeatureSet(MINIMAL_FEATURES).copy()
        fs2 = fs1.copy()
        
        # Should be equal but independent
        assert fs1.feature_names == fs2.feature_names
        
        # Modify one
        fs2.remove_feature("sentiment")
        
        # Original should be unchanged
        assert "sentiment" in fs1.feature_names
        assert "sentiment" not in fs2.feature_names
    
    def test_to_dict_and_from_dict(self):
        """Test converting to/from dict"""
        fs1 = FeatureSet(DEFAULT_FEATURES)
        config = fs1.to_dict()
        
        fs2 = FeatureSet.from_dict(config)
        assert fs1.name == fs2.name
        assert fs1.feature_names == fs2.feature_names
    
    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading from JSON file"""
        fs1 = FeatureSet(MINIMAL_FEATURES)
        
        json_path = tmp_path / "features.json"
        fs1.to_json(str(json_path))
        
        assert json_path.exists()
        
        fs2 = FeatureSet.from_json(str(json_path))
        assert fs1.name == fs2.name
        assert fs1.feature_names == fs2.feature_names
    
    def test_repr(self):
        """Test string representation"""
        fs = FeatureSet(MINIMAL_FEATURES)
        repr_str = repr(fs)
        assert "MinimalFeatures" in repr_str
        assert "n_features=3" in repr_str


class TestCreateFeature:
    """Test the create_feature helper function"""
    
    def test_create_scale_feature(self):
        """Test creating a scale feature"""
        feature = create_feature(
            name="test_scale",
            feature_type="scale",
            description="Test description",
            min_val=1,
            max_val=10
        )
        
        assert feature["name"] == "test_scale"
        assert feature["type"] == "scale"
        assert feature["min"] == 1
        assert feature["max"] == 10
    
    def test_create_bool_feature(self):
        """Test creating a boolean feature"""
        feature = create_feature(
            name="test_bool",
            feature_type="bool",
            description="Test boolean"
        )
        
        assert feature["name"] == "test_bool"
        assert feature["type"] == "bool"
        assert "min" not in feature
        assert "max" not in feature
    
    def test_create_categorical_feature(self):
        """Test creating a categorical feature"""
        feature = create_feature(
            name="test_cat",
            feature_type="categorical",
            description="Test category",
            values=["a", "b", "c"]
        )
        
        assert feature["name"] == "test_cat"
        assert feature["type"] == "categorical"
        assert feature["values"] == ["a", "b", "c"]
    
    def test_create_scale_without_min_max_raises_error(self):
        """Test that creating scale without min/max raises error"""
        with pytest.raises(ValueError, match="require min_val and max_val"):
            create_feature(
                name="bad_scale",
                feature_type="scale",
                description="Missing min/max"
            )
    
    def test_create_categorical_without_values_raises_error(self):
        """Test that creating categorical without values raises error"""
        with pytest.raises(ValueError, match="require values list"):
            create_feature(
                name="bad_cat",
                feature_type="categorical",
                description="Missing values"
            )
    
    def test_create_unknown_type_raises_error(self):
        """Test that unknown feature type raises error"""
        with pytest.raises(ValueError, match="Unknown feature type"):
            create_feature(
                name="bad_type",
                feature_type="invalid",
                description="Bad type"
            )
