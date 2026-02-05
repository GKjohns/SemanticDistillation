"""
Pydantic Schemas for Semantic Distillation

This module handles dynamic Pydantic model generation from feature definitions,
as well as static schemas used in the pipeline.
"""

from typing import Any, Type
from pydantic import BaseModel, Field, create_model


# ---------------------------------------------------------------------------
# Static Schemas (used for LLM responses)
# ---------------------------------------------------------------------------

class FeatureHypothesis(BaseModel):
    """When analyzing residuals, the LLM suggests new features to extract."""
    feature_name: str = Field(..., description="Snake_case name for the proposed feature")
    feature_type: str = Field(..., description="'scale_1_5', 'boolean', or 'categorical'")
    description: str = Field(..., description="What this feature measures and why it might help")
    extraction_prompt: str = Field(..., description="The prompt fragment to extract this feature from text")


class ResidualAnalysis(BaseModel):
    """Analysis of model failures to generate new feature hypotheses."""
    pattern_summary: str = Field(..., description="What patterns exist in the misclassified examples")
    hypotheses: list[FeatureHypothesis] = Field(..., description="3-5 proposed new features")


# ---------------------------------------------------------------------------
# Dynamic Model Generation
# ---------------------------------------------------------------------------

def build_pydantic_model(feature_config: dict[str, Any]) -> Type[BaseModel]:
    """
    Dynamically build a Pydantic model from a feature configuration dict.
    
    Args:
        feature_config: A dict with 'name', 'description', and 'features' keys.
                       See features.py for the expected structure.
    
    Returns:
        A Pydantic BaseModel class with the specified fields.
    
    Example:
        from features import DEFAULT_FEATURES
        from schemas import build_pydantic_model
        
        FeatureModel = build_pydantic_model(DEFAULT_FEATURES)
        # Now FeatureModel can be used for structured LLM extraction
    """
    model_name = feature_config.get("name", "ExtractedFeatures")
    features = feature_config.get("features", [])
    
    field_definitions = {}
    
    for feat in features:
        name = feat["name"]
        ftype = feat["type"]
        desc = feat["description"]
        
        if ftype == "scale":
            min_val = feat.get("min", 1)
            max_val = feat.get("max", 5)
            field_definitions[name] = (
                int,
                Field(..., ge=min_val, le=max_val, description=desc)
            )
        
        elif ftype == "bool":
            field_definitions[name] = (
                bool,
                Field(..., description=desc)
            )
        
        elif ftype == "categorical":
            # For categorical, we use str type
            # The values are documented in the description for the LLM
            values = feat.get("values", [])
            full_desc = f"{desc} (One of: {', '.join(values)})"
            field_definitions[name] = (
                str,
                Field(..., description=full_desc)
            )
        
        else:
            raise ValueError(f"Unknown feature type: {ftype} for feature {name}")
    
    # Create the model dynamically
    model = create_model(
        model_name,
        __doc__=feature_config.get("description", "Extracted features from text."),
        **field_definitions
    )
    
    return model


def get_feature_field_types(feature_config: dict[str, Any]) -> dict[str, str]:
    """
    Get a mapping of feature names to their types.
    
    Returns:
        Dict mapping feature name to 'scale', 'bool', or 'categorical'
    """
    return {f["name"]: f["type"] for f in feature_config.get("features", [])}


def get_categorical_values(feature_config: dict[str, Any]) -> dict[str, list[str]]:
    """
    Get the allowed values for categorical features.
    
    Returns:
        Dict mapping categorical feature names to their allowed values
    """
    return {
        f["name"]: f.get("values", [])
        for f in feature_config.get("features", [])
        if f["type"] == "categorical"
    }
