"""
Semantic Distillation Package

LLM-powered interpretable feature engineering.
"""

from .features import (
    DEFAULT_FEATURES,
    MINIMAL_FEATURES,
    FeatureSet,
    create_feature,
)
from .schemas import (
    build_pydantic_model,
    ResidualAnalysis,
    FeatureHypothesis,
)
from .utils import (
    setup_logging,
    FeatureCache,
    RateLimiter,
    save_results,
    load_results,
    save_features_csv,
)
from .data import (
    SAMPLE_AITA_POSTS,
    get_sample_aita_data,
    load_data,
)
from .semantic_distillation import SemanticDistiller

__all__ = [
    # Features
    "DEFAULT_FEATURES",
    "MINIMAL_FEATURES",
    "FeatureSet",
    "create_feature",
    # Schemas
    "build_pydantic_model",
    "ResidualAnalysis",
    "FeatureHypothesis",
    # Utils
    "setup_logging",
    "FeatureCache",
    "RateLimiter",
    "save_results",
    "load_results",
    "save_features_csv",
    # Data
    "SAMPLE_AITA_POSTS",
    "get_sample_aita_data",
    "load_data",
    # Core
    "SemanticDistiller",
]
