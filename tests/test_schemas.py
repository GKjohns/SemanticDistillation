"""
Tests for schemas.py - Pydantic model generation and validation
"""

import pytest
from pydantic import ValidationError
from src.schemas import (
    FeatureHypothesis,
    ResidualAnalysis,
    build_pydantic_model,
    get_feature_field_types,
    get_categorical_values,
)
from src.features import DEFAULT_FEATURES, MINIMAL_FEATURES


class TestStaticSchemas:
    """Test static Pydantic schemas"""
    
    def test_feature_hypothesis_valid(self):
        """Test creating a valid FeatureHypothesis"""
        hyp = FeatureHypothesis(
            feature_name="test_feature",
            feature_type="scale_1_5",
            description="A test feature",
            extraction_prompt="Rate this on a scale of 1-5"
        )
        assert hyp.feature_name == "test_feature"
        assert hyp.feature_type == "scale_1_5"
    
    def test_feature_hypothesis_missing_field(self):
        """Test that missing required fields raise ValidationError"""
        with pytest.raises(ValidationError):
            FeatureHypothesis(
                feature_name="test",
                feature_type="scale_1_5"
                # Missing description and extraction_prompt
            )
    
    def test_residual_analysis_valid(self):
        """Test creating a valid ResidualAnalysis"""
        analysis = ResidualAnalysis(
            pattern_summary="Some pattern found",
            hypotheses=[
                FeatureHypothesis(
                    feature_name="test1",
                    feature_type="boolean",
                    description="Test feature 1",
                    extraction_prompt="Is this true or false?"
                )
            ]
        )
        assert analysis.pattern_summary == "Some pattern found"
        assert len(analysis.hypotheses) == 1


class TestDynamicModelGeneration:
    """Test dynamic Pydantic model generation from feature configs"""
    
    def test_build_model_with_default_features(self):
        """Test building a model from DEFAULT_FEATURES"""
        Model = build_pydantic_model(DEFAULT_FEATURES)
        assert Model.__name__ == "AITAFeatures"
        
        # Check that all feature fields exist
        fields = Model.model_fields
        assert "self_awareness" in fields
        assert "empathy_shown" in fields
        assert "harm_caused" in fields
    
    def test_build_model_with_minimal_features(self):
        """Test building a model from MINIMAL_FEATURES"""
        Model = build_pydantic_model(MINIMAL_FEATURES)
        assert Model.__name__ == "MinimalFeatures"
        
        fields = Model.model_fields
        assert "sentiment" in fields
        assert "self_aware" in fields
        assert "conflict_type" in fields
    
    def test_scale_field_validation(self):
        """Test that scale fields validate min/max constraints"""
        Model = build_pydantic_model({
            "name": "TestModel",
            "features": [
                {
                    "name": "test_scale",
                    "type": "scale",
                    "min": 1,
                    "max": 5,
                    "description": "A test scale"
                }
            ]
        })
        
        # Valid value
        instance = Model(test_scale=3)
        assert instance.test_scale == 3
        
        # Invalid value - too high
        with pytest.raises(ValidationError):
            Model(test_scale=6)
        
        # Invalid value - too low
        with pytest.raises(ValidationError):
            Model(test_scale=0)
    
    def test_bool_field_validation(self):
        """Test that bool fields work correctly"""
        Model = build_pydantic_model({
            "name": "TestModel",
            "features": [
                {
                    "name": "test_bool",
                    "type": "bool",
                    "description": "A test boolean"
                }
            ]
        })
        
        instance_true = Model(test_bool=True)
        assert instance_true.test_bool is True
        
        instance_false = Model(test_bool=False)
        assert instance_false.test_bool is False
    
    def test_categorical_field_validation(self):
        """Test that categorical fields work correctly"""
        Model = build_pydantic_model({
            "name": "TestModel",
            "features": [
                {
                    "name": "test_category",
                    "type": "categorical",
                    "values": ["red", "green", "blue"],
                    "description": "A test category"
                }
            ]
        })
        
        # All values should be accepted (validation happens at extraction time)
        instance = Model(test_category="red")
        assert instance.test_category == "red"
    
    def test_unknown_feature_type_raises_error(self):
        """Test that unknown feature types raise ValueError"""
        with pytest.raises(ValueError, match="Unknown feature type"):
            build_pydantic_model({
                "name": "TestModel",
                "features": [
                    {
                        "name": "bad_feature",
                        "type": "invalid_type",
                        "description": "This should fail"
                    }
                ]
            })


class TestHelperFunctions:
    """Test helper functions for feature configs"""
    
    def test_get_feature_field_types(self):
        """Test extracting feature types from config"""
        types = get_feature_field_types(MINIMAL_FEATURES)
        assert types["sentiment"] == "scale"
        assert types["self_aware"] == "bool"
        assert types["conflict_type"] == "categorical"
    
    def test_get_categorical_values(self):
        """Test extracting categorical values from config"""
        values = get_categorical_values(MINIMAL_FEATURES)
        assert "conflict_type" in values
        assert "personal" in values["conflict_type"]
        assert "professional" in values["conflict_type"]
    
    def test_get_categorical_values_empty_when_no_categoricals(self):
        """Test that get_categorical_values returns empty dict when no categoricals"""
        config = {
            "features": [
                {"name": "test", "type": "scale", "min": 1, "max": 5, "description": "test"}
            ]
        }
        values = get_categorical_values(config)
        assert values == {}
