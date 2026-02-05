"""
Feature Definitions for Semantic Distillation

This module defines the feature sets used for extraction. Features are defined
as configuration dictionaries that can be easily modified, extended, or swapped.

Each feature definition includes:
- name: The field name (snake_case)
- type: One of 'scale', 'bool', or 'categorical'
- description: What the feature measures (used in the LLM prompt)
- For scale types: min/max values (typically 1-5)
- For categorical types: allowed values list

Usage:
    from features import DEFAULT_FEATURES, FeatureSet
    
    # Use defaults
    feature_set = FeatureSet(DEFAULT_FEATURES)
    
    # Or customize
    my_features = DEFAULT_FEATURES.copy()
    my_features['features'].append({...})
    feature_set = FeatureSet(my_features)
"""

from typing import Any
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Default AITA Feature Definitions
# ---------------------------------------------------------------------------

DEFAULT_FEATURES: dict[str, Any] = {
    "name": "AITAFeatures",
    "description": "Features extracted from an AITA post. Each feature is designed to be interpretable and blind to the outcome (no leakage). All features are continuous scales to enable meaningful coefficient interpretation.",
    "features": [
        # Narrative & self-presentation
        {
            "name": "self_awareness",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=no acknowledgment of own role, 5=deep reflection on own behavior and potential wrongdoing",
        },
        {
            "name": "empathy_shown",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=zero empathy for the other party, 5=deeply empathetic and understanding of their perspective",
        },
        {
            "name": "emotional_intensity",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=calm/neutral tone, 5=highly emotional (angry, upset, defensive)",
        },
        
        # Actions & harm
        {
            "name": "harm_caused",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=no harm or very minor inconvenience, 5=serious harm (financial, emotional, physical, relational)",
        },
        {
            "name": "provocation_received",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=poster acted unprovoked, 5=poster was severely provoked or wronged first",
        },
        {
            "name": "proportionality",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=response was way out of proportion to the situation, 5=response was measured and appropriate",
        },
        
        # Relationship dynamics
        {
            "name": "relationship_closeness",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=strangers/acquaintances, 5=very close relationship (spouse, parent, best friend)",
        },
        {
            "name": "power_dynamic",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=poster has more power (boss, parent, host), 3=equal, 5=other party has more power",
        },
    ],
}


# Full feature set with all 15 features
FULL_FEATURES: dict[str, Any] = {
    "name": "AITAFeaturesFull",
    "description": "Comprehensive features extracted from an AITA post. Each feature is designed to be interpretable and blind to the outcome (no leakage).",
    "features": [
        # Narrative structure
        {
            "name": "perspective_balance",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=completely one-sided, 5=thoroughly presents both perspectives",
        },
        {
            "name": "self_awareness",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=no acknowledgment of own role, 5=deep reflection on own behavior",
        },
        {
            "name": "minimization",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=no minimization of own actions, 5=heavily downplays what they did",
        },
        
        # Conflict characteristics
        {
            "name": "power_imbalance",
            "type": "bool",
            "description": "Is there a clear power imbalance between parties (boss/employee, parent/child, etc)?",
        },
        {
            "name": "financial_component",
            "type": "bool",
            "description": "Does the conflict involve money, property, or financial obligations?",
        },
        {
            "name": "children_involved",
            "type": "bool",
            "description": "Are children (minors) directly affected by or involved in the conflict?",
        },
        {
            "name": "betrayal_of_trust",
            "type": "bool",
            "description": "Does the conflict involve a betrayal of trust, broken promise, or violation of an agreement?",
        },
        {
            "name": "boundary_setting",
            "type": "bool",
            "description": "Is the poster primarily trying to set or enforce a personal boundary?",
        },
        
        # Emotional tone
        {
            "name": "anger_level",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=calm/neutral, 5=extremely angry or hostile tone",
        },
        {
            "name": "guilt_expressed",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=no guilt whatsoever, 5=clearly wracked with guilt about what happened",
        },
        {
            "name": "empathy_shown",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=zero empathy for the other party, 5=deeply empathetic even while disagreeing",
        },
        
        # Situational
        {
            "name": "relationship_type",
            "type": "categorical",
            "values": ["romantic", "family", "friendship", "workplace", "neighbor", "stranger", "other"],
            "description": "Primary relationship: 'romantic', 'family', 'friendship', 'workplace', 'neighbor', 'stranger', 'other'",
        },
        {
            "name": "escalation_initiated",
            "type": "bool",
            "description": "Did the poster escalate the conflict (vs respond to someone else's escalation)?",
        },
        {
            "name": "social_norm_violation",
            "type": "bool",
            "description": "Does the post describe a violation of widely-held social norms or etiquette?",
        },
        {
            "name": "revenge_or_retaliation",
            "type": "bool",
            "description": "Is the poster's action primarily motivated by revenge or retaliation?",
        },
    ],
}


# ---------------------------------------------------------------------------
# Alternative Feature Sets (examples)
# ---------------------------------------------------------------------------

# A minimal feature set for testing or simpler analysis
MINIMAL_FEATURES: dict[str, Any] = {
    "name": "MinimalFeatures",
    "description": "A minimal set of features for quick analysis.",
    "features": [
        {
            "name": "sentiment",
            "type": "scale",
            "min": 1,
            "max": 5,
            "description": "1=very negative tone, 5=very positive tone",
        },
        {
            "name": "self_aware",
            "type": "bool",
            "description": "Does the poster show awareness of their own potential wrongdoing?",
        },
        {
            "name": "conflict_type",
            "type": "categorical",
            "values": ["personal", "professional", "family", "other"],
            "description": "The primary domain of the conflict",
        },
    ],
}


# ---------------------------------------------------------------------------
# FeatureSet class for working with feature definitions
# ---------------------------------------------------------------------------

@dataclass
class FeatureSet:
    """
    A wrapper around feature definitions that provides utility methods.
    
    Usage:
        fs = FeatureSet(DEFAULT_FEATURES)
        fs.add_feature({...})
        fs.remove_feature("some_feature")
        pydantic_model = fs.to_pydantic_model()
    """
    
    config: dict[str, Any]
    
    @property
    def name(self) -> str:
        return self.config.get("name", "ExtractedFeatures")
    
    @property
    def description(self) -> str:
        return self.config.get("description", "")
    
    @property
    def features(self) -> list[dict[str, Any]]:
        return self.config.get("features", [])
    
    @property
    def feature_names(self) -> list[str]:
        return [f["name"] for f in self.features]
    
    def add_feature(self, feature: dict[str, Any]) -> None:
        """Add a feature to the set."""
        if feature["name"] in self.feature_names:
            raise ValueError(f"Feature '{feature['name']}' already exists")
        self.config["features"].append(feature)
    
    def remove_feature(self, name: str) -> None:
        """Remove a feature by name."""
        self.config["features"] = [f for f in self.features if f["name"] != name]
    
    def get_feature(self, name: str) -> dict[str, Any] | None:
        """Get a feature definition by name."""
        for f in self.features:
            if f["name"] == name:
                return f
        return None
    
    def copy(self) -> "FeatureSet":
        """Create a deep copy of this feature set."""
        import copy
        return FeatureSet(copy.deepcopy(self.config))
    
    def to_dict(self) -> dict[str, Any]:
        """Return the underlying config dict."""
        return self.config
    
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FeatureSet":
        """Create a FeatureSet from a config dict."""
        return cls(config)
    
    @classmethod
    def from_json(cls, path: str) -> "FeatureSet":
        """Load a FeatureSet from a JSON file."""
        import json
        from pathlib import Path
        config = json.loads(Path(path).read_text())
        return cls(config)
    
    def to_json(self, path: str) -> None:
        """Save this FeatureSet to a JSON file."""
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.config, indent=2))
    
    def __repr__(self) -> str:
        return f"FeatureSet(name={self.name!r}, n_features={len(self.features)})"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def create_feature(
    name: str,
    feature_type: str,
    description: str,
    *,
    min_val: int | None = None,
    max_val: int | None = None,
    values: list[str] | None = None,
) -> dict[str, Any]:
    """
    Helper to create a feature definition dict.
    
    Args:
        name: Feature name (snake_case)
        feature_type: One of 'scale', 'bool', 'categorical'
        description: What the feature measures
        min_val: For scale type, the minimum value
        max_val: For scale type, the maximum value
        values: For categorical type, the allowed values
    
    Returns:
        Feature definition dict
    """
    feature = {
        "name": name,
        "type": feature_type,
        "description": description,
    }
    
    if feature_type == "scale":
        if min_val is None or max_val is None:
            raise ValueError("Scale features require min_val and max_val")
        feature["min"] = min_val
        feature["max"] = max_val
    elif feature_type == "categorical":
        if values is None:
            raise ValueError("Categorical features require values list")
        feature["values"] = values
    elif feature_type != "bool":
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return feature
