"""
Iterative Feature Improvement for Semantic Distillation

This script automates the process of improving the feature set used for
semantic distillation. It:
1. Runs the distillation pipeline with current features
2. Analyzes the results to understand what's working
3. Uses an LLM to evaluate features and propose improvements
4. Swaps out underperforming features for new ones
5. Repeats to iteratively improve accuracy

Usage:
    python iterative_improvement.py --iterations 5
    python iterative_improvement.py --iterations 3 --max-swaps 4 --sample-size 50
    
    # With validation split (recommended for robust feature selection):
    python iterative_improvement.py --iterations 5 --validation-split
"""

import json
import copy
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support


# ---------------------------------------------------------------------------
# Verbose Logging Setup
# ---------------------------------------------------------------------------

class VerboseLogger:
    """
    Writes a detailed, human-readable log of the iterative improvement process.
    
    Tracks model decisions, reasoning, and evaluation stats at each iteration
    without logging every individual labeling operation.
    """
    
    def __init__(self, output_dir: Path, session_id: str):
        self.output_dir = output_dir
        self.session_id = session_id
        self.log_path = output_dir / f"verbose_log_{session_id}.md"
        
        # Initialize the log file with header
        self._write_header()
    
    def _write(self, text: str, newline: bool = True):
        """Append text to the log file."""
        with open(self.log_path, "a") as f:
            f.write(text)
            if newline:
                f.write("\n")
    
    def _write_header(self):
        """Write the log file header."""
        with open(self.log_path, "w") as f:
            f.write(f"# Iterative Feature Improvement Log\n\n")
            f.write(f"**Session ID:** `{self.session_id}`\n")
            f.write(f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
    
    def log_initialization(
        self,
        data_path: str,
        sample_size: int,
        max_swaps: int,
        analysis_model: str,
        extraction_model: str,
        validation_split: bool,
        train_pool_size: int = None,
        validation_size: int = None,
        test_size: int = None,
        n_folds: int = None,
        initial_features: list[str] = None,
    ):
        """Log initialization parameters."""
        self._write("## Configuration\n")
        self._write(f"- **Data path:** `{data_path}`")
        self._write(f"- **Sample size per iteration:** {sample_size}")
        self._write(f"- **Max feature swaps per iteration:** {max_swaps}")
        self._write(f"- **Analysis model:** `{analysis_model}`")
        self._write(f"- **Extraction model:** `{extraction_model}`")
        
        if validation_split:
            self._write(f"- **Evaluation strategy:** Train/Validation/Test Split")
            self._write(f"  - Train pool: {train_pool_size} samples")
            self._write(f"  - Validation: {validation_size} samples (fixed)")
            self._write(f"  - Test: {test_size} samples (held out)")
            self._write(f"  - Train folds: {n_folds}")
        else:
            self._write(f"- **Evaluation strategy:** Single sample (legacy)")
        
        if initial_features:
            self._write(f"\n### Initial Feature Set ({len(initial_features)} features)\n")
            for feat in initial_features:
                self._write(f"- `{feat}`")
        
        self._write("\n---\n")
    
    def log_iteration_start(self, iteration_num: int, features: list[dict]):
        """Log the start of an iteration."""
        self._write(f"## Iteration {iteration_num}\n")
        self._write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self._write("### Current Feature Set\n")
        for f in features:
            feat_type = f.get("type", "unknown")
            if feat_type == "scale":
                range_info = f" (scale: {f.get('min', 1)}-{f.get('max', 5)})"
            elif feat_type == "categorical":
                values = f.get("values", [])
                range_info = f" (categorical: {', '.join(values[:3])}{'...' if len(values) > 3 else ''})"
            else:
                range_info = f" ({feat_type})"
            self._write(f"- **`{f['name']}`**{range_info}: {f.get('description', 'N/A')}")
        self._write("")
    
    def log_evaluation_results(
        self,
        train_cv_accuracy: float,
        train_cv_std: float,
        val_accuracy: float = None,
        n_train: int = None,
        n_val: int = None,
        n_nonzero_features: int = None,
        total_features: int = None,
        regularization_c: float = None,
        is_best: bool = False,
    ):
        """Log evaluation metrics."""
        self._write("### Evaluation Results\n")
        
        self._write(f"| Metric | Value |")
        self._write(f"|--------|-------|")
        self._write(f"| Train CV Accuracy | **{train_cv_accuracy:.3f}** ± {train_cv_std:.3f} |")
        
        if val_accuracy is not None:
            marker = " ⭐ (new best)" if is_best else ""
            self._write(f"| Validation Accuracy | **{val_accuracy:.3f}**{marker} |")
        
        if n_train is not None:
            self._write(f"| Train Samples | {n_train} |")
        if n_val is not None:
            self._write(f"| Validation Samples | {n_val} |")
        if n_nonzero_features is not None and total_features is not None:
            self._write(f"| Active Features (post-L1) | {n_nonzero_features}/{total_features} |")
        if regularization_c is not None:
            self._write(f"| Selected Regularization C | {regularization_c:.4f} |")
        
        self._write("")
    
    def log_coefficients(self, coefficients: list[dict], top_n: int = 10):
        """Log top feature coefficients."""
        self._write("### Feature Coefficients (Top by Magnitude)\n")
        
        sorted_coefs = sorted(coefficients, key=lambda x: -abs(x["coefficient"]))[:top_n]
        
        self._write("| Feature | Coefficient | Odds Ratio | Direction |")
        self._write("|---------|-------------|------------|-----------|")
        for c in sorted_coefs:
            coef = c["coefficient"]
            direction = "↑ (positive)" if coef > 0 else "↓ (negative)" if coef < 0 else "neutral"
            self._write(f"| `{c['feature']}` | {coef:+.4f} | {c['odds_ratio']:.3f} | {direction} |")
        
        self._write("")
    
    def log_per_class_metrics(self, class_metrics: list[dict]):
        """Log precision, recall, and support for each class."""
        self._write("### Per-Class Performance\n")
        
        self._write("| Class | Precision | Recall | F1-Score | Support (n) |")
        self._write("|-------|-----------|--------|----------|-------------|")
        for m in class_metrics:
            self._write(
                f"| **{m['class']}** | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1']:.3f} | {m['support']} |"
            )
        
        self._write("")
    
    def log_sample_analysis(
        self,
        correct_patterns: str,
        incorrect_patterns: str,
        missing_signals: list[str],
        key_insight: str,
    ):
        """Log sample analysis insights from the LLM."""
        self._write("### Sample Analysis (LLM Insights)\n")
        
        self._write("**Patterns in Correctly Classified Samples:**")
        self._write(f"> {correct_patterns}\n")
        
        self._write("**Patterns in Misclassified Samples:**")
        self._write(f"> {incorrect_patterns}\n")
        
        self._write("**Missing Signals Identified:**")
        for signal in missing_signals:
            self._write(f"- {signal}")
        self._write("")
        
        self._write("**Key Insight:**")
        self._write(f"> 💡 {key_insight}\n")
    
    def log_broad_patterns(self, patterns: list[str], sample_size: int):
        """Log one-time broad pattern discovery output."""
        self._write("## Broad Pattern Discovery (Pre-Loop)\n")
        self._write(
            f"Ran broad pattern discovery on **{sample_size}** posts (text + verdict only). "
            "These are intended as stable domain priors to ground per-iteration error analysis."
        )
        self._write("")
        for i, p in enumerate(patterns, start=1):
            self._write(f"{i}. {p}")
        self._write("\n---\n")
    
    def log_feature_evaluations(self, evaluations: list[dict]):
        """Log individual feature evaluations."""
        self._write("### Feature Evaluations\n")
        
        # Sort by usefulness score (low to high to highlight weak features)
        sorted_evals = sorted(evaluations, key=lambda x: x["usefulness_score"])
        
        for eval in sorted_evals:
            score = eval["usefulness_score"]
            stars = "⭐" * score + "☆" * (5 - score)
            keep_status = "✅ Keep" if eval["keep"] else "❌ Remove"
            
            self._write(f"#### `{eval['feature_name']}` - {stars} ({score}/5) - {keep_status}\n")
            self._write(f"> {eval['reasoning']}\n")
    
    def log_feature_changes(
        self,
        features_removed: list[str],
        features_added: list[dict],
        expected_improvement: str,
    ):
        """Log feature swap decisions."""
        self._write("### Feature Changes Decision\n")
        
        if features_removed:
            self._write("**Features Removed:**")
            for feat in features_removed:
                self._write(f"- ❌ `{feat}`")
            self._write("")
        
        if features_added:
            self._write("**Features Added:**")
            for feat in features_added:
                name = feat.get("name", feat.get("feature_name", "unknown"))
                feat_type = feat.get("feature_type", feat.get("type", "unknown"))
                description = feat.get("description", "N/A")
                rationale = feat.get("rationale", "N/A")
                
                self._write(f"- ✅ **`{name}`** ({feat_type})")
                self._write(f"  - *Description:* {description}")
                self._write(f"  - *Rationale:* {rationale}")
            self._write("")
        
        self._write("**Expected Improvement:**")
        self._write(f"> {expected_improvement}\n")
    
    def log_iteration_summary(self, summary: str):
        """Log the LLM's summary of the iteration."""
        self._write("### Iteration Summary\n")
        self._write(f"> {summary}\n")
        self._write("\n---\n")
    
    def log_test_evaluation(
        self,
        test_accuracy: float,
        n_test: int,
        best_val_accuracy: float,
        per_class_metrics: list[dict] = None,
    ):
        """Log final test set evaluation."""
        self._write("## Final Test Set Evaluation\n")
        self._write(f"**Test Accuracy:** **{test_accuracy:.3f}**")
        self._write(f"**Test Samples:** {n_test}")
        self._write(f"**Best Validation Accuracy:** {best_val_accuracy:.3f}\n")
        
        if per_class_metrics:
            self._write("### Per-Class Test Performance\n")
            self._write("| Class | Precision | Recall | F1-Score | Support (n) |")
            self._write("|-------|-----------|--------|----------|-------------|")
            for m in per_class_metrics:
                self._write(
                    f"| **{m['class']}** | {m['precision']:.3f} | {m['recall']:.3f} | "
                    f"{m['f1']:.3f} | {m['support']} |"
                )
            self._write("")
        
        self._write(f"*Note: Test set was held out and never used during feature selection.*\n")
        self._write("\n---\n")
    
    def log_final_summary(
        self,
        history: list[dict],
        best_iteration: int,
        best_accuracy: float,
        best_features: list[str],
        validation_split: bool,
    ):
        """Log the final summary of all iterations."""
        self._write("## Final Summary\n")
        self._write(f"**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Accuracy progression table
        self._write("### Accuracy Progression\n")
        
        if validation_split:
            self._write("| Iteration | Train Fold | Train CV | Val Accuracy | Features Changed |")
            self._write("|-----------|------------|----------|--------------|------------------|")
            for r in history:
                removed = ", ".join(r["features_removed"][:2])
                if len(r["features_removed"]) > 2:
                    removed += f" +{len(r['features_removed'])-2}"
                added_names = [f.get("name", f.get("feature_name", "?")) for f in r["features_added"]]
                added = ", ".join(added_names[:2])
                if len(r["features_added"]) > 2:
                    added += f" +{len(r['features_added'])-2}"
                changes = f"-({removed}), +({added})" if removed else "baseline"
                
                self._write(
                    f"| {r['iteration']} | {r.get('train_fold', 0)+1} | "
                    f"{r['train_cv_accuracy']:.3f}±{r['train_cv_std']:.2f} | "
                    f"{r['val_accuracy']:.3f} | {changes} |"
                )
        else:
            self._write("| Iteration | CV Accuracy | Train Accuracy | Features Changed |")
            self._write("|-----------|-------------|----------------|------------------|")
            for r in history:
                removed = ", ".join(r["features_removed"][:2])
                added_names = [f.get("name", f.get("feature_name", "?")) for f in r["features_added"]]
                added = ", ".join(added_names[:2])
                changes = f"-({removed}), +({added})" if removed else "baseline"
                
                self._write(
                    f"| {r['iteration']} | {r['cv_accuracy']:.3f}±{r['cv_std']:.2f} | "
                    f"{r.get('train_accuracy', 'N/A'):.3f} | {changes} |"
                )
        
        self._write("")
        
        # Improvement summary
        if len(history) > 1:
            if validation_split:
                first_acc = history[0]["val_accuracy"]
                last_acc = history[-1]["val_accuracy"]
            else:
                first_acc = history[0]["cv_accuracy"]
                last_acc = history[-1]["cv_accuracy"]
            
            improvement = last_acc - first_acc
            trend = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            self._write(f"### Overall Progress {trend}\n")
            self._write(f"- **Starting accuracy:** {first_acc:.3f}")
            self._write(f"- **Ending accuracy:** {last_acc:.3f}")
            self._write(f"- **Net change:** {improvement:+.3f}")
        
        self._write(f"\n### Best Configuration\n")
        self._write(f"- **Best iteration:** {best_iteration}")
        self._write(f"- **Best accuracy:** {best_accuracy:.3f}")
        self._write(f"- **Final feature set:**")
        for feat in best_features:
            self._write(f"  - `{feat}`")
        
        self._write("\n---\n")
        self._write(f"*Log file: `{self.log_path}`*")

# Local imports - need to be in path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from semantic_distillation import SemanticDistiller
from features import DEFAULT_FEATURES
from data import load_data


# ---------------------------------------------------------------------------
# Pydantic Schemas for LLM Analysis
# ---------------------------------------------------------------------------

class FeatureEvaluation(BaseModel):
    """Evaluation of a single feature's usefulness."""
    feature_name: str = Field(..., description="Name of the feature being evaluated")
    usefulness_score: int = Field(..., ge=1, le=5, description="1=useless/harmful, 5=very predictive")
    reasoning: str = Field(..., description="Why this feature is or isn't useful")
    keep: bool = Field(..., description="Whether to keep this feature in the next iteration")


class ProposedFeature(BaseModel):
    """A new feature proposed to replace an underperforming one."""
    name: str = Field(..., description="Snake_case name for the feature")
    feature_type: str = Field(..., description="'scale', 'bool', or 'categorical'")
    description: str = Field(..., description="What this feature measures (will be used in extraction prompt)")
    min_val: Optional[int] = Field(None, description="For scale type: minimum value")
    max_val: Optional[int] = Field(None, description="For scale type: maximum value")
    values: Optional[list[str]] = Field(None, description="For categorical type: allowed values")
    rationale: str = Field(..., description="Why this feature should help classification")


class IterationAnalysis(BaseModel):
    """Complete analysis of an iteration's results."""
    summary: str = Field(..., description="Brief summary of model performance this iteration")
    feature_evaluations: list[FeatureEvaluation] = Field(..., description="Evaluation of each feature")
    features_to_remove: list[str] = Field(..., description="Names of features to swap out (up to max_swaps)")
    proposed_features: list[ProposedFeature] = Field(..., description="New features to add (variable count)")
    expected_improvement: str = Field(..., description="What improvement we expect from these changes")


class SampleAnalysis(BaseModel):
    """Analysis of correctly and incorrectly classified samples."""
    correct_patterns: str = Field(..., description="What patterns exist in correctly classified samples")
    incorrect_patterns: str = Field(..., description="What patterns exist in misclassified samples")
    missing_signals: list[str] = Field(..., description="Signals that could help but aren't captured")
    key_insight: str = Field(..., description="The most important insight for improvement")

class BroadPatternDiscovery(BaseModel):
    """High-level patterns that distinguish verdicts (run once per session)."""
    patterns: list[str] = Field(
        ...,
        description="10-15 general patterns that distinguish NTA vs YTA posts; concrete but not overly specific to any single example.",
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def estimate_feature_effective_dim(feature: dict) -> int:
    """
    Estimate effective dimensionality contributed by a single raw feature after encoding.
    
    - bool/scale: 1 column
    - categorical: (k - 1) columns due to one-hot encoding with drop_first=True
    
    Note: this is an *upper bound*; in any given fold, unobserved categories may
    produce fewer columns.
    """
    ftype = feature.get("type") or feature.get("feature_type")
    if ftype == "categorical":
        values = feature.get("values") or []
        return max(0, len(values) - 1)
    # scale / bool / unknown: assume 1
    return 1


def estimate_config_effective_dim(feature_config: dict) -> int:
    """Estimate total effective dimensionality for a feature config."""
    feats = feature_config.get("features") or []
    return int(sum(estimate_feature_effective_dim(f) for f in feats))


def compute_dimensionality_budget(n_train: int) -> int:
    """
    Compute a hard cap on effective feature dimensionality (post one-hot).
    
    Heuristic: ~0.25 * n_train, clamped to [12, 25].
    - n_train=74 -> 18
    - n_train=50 -> 12
    """
    return int(max(12, min(25, round(0.25 * n_train))))


def top_feature_column_costs(
    X: pd.DataFrame,
    raw_features: pd.DataFrame,
    top_k: int = 10,
) -> list[tuple[str, int]]:
    """
    Given a post-encoding design matrix X and the raw extracted features dataframe,
    estimate how many encoded columns each raw feature contributes.
    """
    costs: dict[str, int] = {}
    for col in raw_features.columns:
        # categorical features become <col>_<value> dummy columns
        if pd.api.types.is_object_dtype(raw_features[col]) or pd.api.types.is_string_dtype(raw_features[col]):
            prefix = f"{col}_"
            costs[col] = int(sum(1 for c in X.columns if str(c).startswith(prefix)))
        else:
            costs[col] = int(1 if col in X.columns else 0)

    return sorted(costs.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]


def enforce_dimensionality_budget_on_config(
    feature_config: dict,
    budget: int,
) -> tuple[dict, dict]:
    """
    Enforce an effective dimensionality budget on a *feature config* by pruning
    the highest-cost features (categoricals first).
    
    Returns (new_config, info) where info contains any auto removals.
    """
    new_config = copy.deepcopy(feature_config)
    feats = list(new_config.get("features") or [])
    
    def cost(f: dict) -> int:
        return estimate_feature_effective_dim(f)
    
    info = {
        "auto_removed_for_budget": [],  # feature names
        "estimated_dim_before": estimate_config_effective_dim({"features": feats}),
        "estimated_dim_after": None,
        "budget": int(budget),
    }
    
    # Remove highest-cost features until within budget.
    # Prefer removing categoricals, then scales, then bools.
    def removal_priority(f: dict) -> tuple[int, int]:
        # higher is removed first
        ftype = f.get("type", "unknown")
        ftype_rank = 2 if ftype == "categorical" else 1 if ftype == "scale" else 0
        return (ftype_rank, cost(f))
    
    while estimate_config_effective_dim({"features": feats}) > budget and feats:
        feats_sorted = sorted(feats, key=lambda f: (removal_priority(f)[0], removal_priority(f)[1]), reverse=True)
        to_remove = feats_sorted[0]
        feats = [f for f in feats if f is not to_remove]
        info["auto_removed_for_budget"].append(to_remove.get("name", "unknown"))
    
    new_config["features"] = feats
    info["estimated_dim_after"] = estimate_config_effective_dim(new_config)
    return new_config, info


def get_predictions_with_details(
    features_df: pd.DataFrame,
    labels: pd.Series,
    texts: pd.Series,
) -> pd.DataFrame:
    """
    Get detailed predictions including probabilities and correctness.
    
    Returns DataFrame with: features, true_label, predicted_label, correct, confidence
    """
    X, feature_names = SemanticDistiller.build_feature_matrix(features_df)
    
    # Ensure consistent index types (convert to string for comparison)
    X.index = X.index.astype(str)
    labels.index = labels.index.astype(str)
    texts.index = texts.index.astype(str)
    features_df.index = features_df.index.astype(str)
    
    # Align data
    common_idx = X.index.intersection(labels.index).intersection(texts.index)
    if len(common_idx) == 0:
        raise ValueError("No common indices between features and labels - check data alignment")
    
    X = X.loc[common_idx]
    y = labels.loc[common_idx]
    texts = texts.loc[common_idx]
    
    # Encode labels
    label_map = {label: i for i, label in enumerate(sorted(y.unique()))}
    reverse_map = {v: k for k, v in label_map.items()}
    y_enc = y.map(label_map)
    
    # Guard: need at least 2 classes to fit logistic regression
    if len(label_map) < 2:
        majority_label = y.value_counts().idxmax()
        results = pd.DataFrame({
            "text": texts.values,
            "true_label": y.values,
            "predicted_label": [majority_label] * len(y),
            "correct": [True] * len(y),
            "confidence": [1.0] * len(y),
        }, index=common_idx)
        for col in features_df.columns:
            results[f"feat_{col}"] = features_df.loc[common_idx, col].values
        return results
    
    # Fit model and get predictions
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5, solver="saga", C=1.0, max_iter=5000,
            class_weight="balanced", random_state=42
        )),
    ])
    pipe.fit(X, y_enc)
    
    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)
    
    # Build results DataFrame
    results = pd.DataFrame({
        "text": texts.values,
        "true_label": y.values,
        "predicted_label": [reverse_map[p] for p in preds],
        "correct": preds == y_enc.values,
        "confidence": probs.max(axis=1),
    }, index=common_idx)
    
    # Add feature values
    for col in features_df.columns:
        results[f"feat_{col}"] = features_df.loc[common_idx, col].values
    
    return results


def format_sample_for_prompt(row: pd.Series, max_text_len: int = 400) -> str:
    """Format a sample for inclusion in an LLM prompt."""
    text = row["text"][:max_text_len]
    if len(row["text"]) > max_text_len:
        text += "..."
    
    features = {k.replace("feat_", ""): v for k, v in row.items() if k.startswith("feat_")}
    
    return f"""---
TRUE: {row['true_label']} | PREDICTED: {row['predicted_label']} | CONFIDENCE: {row['confidence']:.2f}
TEXT: {text}
FEATURES: {json.dumps(features, default=str)}
"""


def feature_config_to_dict(feature: dict) -> dict:
    """Convert a ProposedFeature-style dict to feature config format."""
    result = {
        "name": feature["name"],
        "type": feature["feature_type"],
        "description": feature["description"],
    }
    
    if feature["feature_type"] == "scale":
        result["min"] = feature.get("min_val", 1)
        result["max"] = feature.get("max_val", 5)
    elif feature["feature_type"] == "categorical":
        result["values"] = feature.get("values", [])
    
    return result


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> list[dict]:
    """
    Compute precision, recall, f1, and support for each class.
    
    Returns list of dicts with keys: class, precision, recall, f1, support
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    metrics = []
    for i, class_name in enumerate(class_names):
        metrics.append({
            "class": class_name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })
    
    return metrics


# ---------------------------------------------------------------------------
# Core Iteration Logic
# ---------------------------------------------------------------------------

class IterativeImprover:
    """Manages the iterative feature improvement process."""
    
    def __init__(
        self,
        data_path: str = "dataset/aita_dataset.csv",
        sample_size: int = 50,
        max_swaps: int = 4,
        analysis_model: str = "gpt-5.2",  # Use powerful model for strategic decisions
        extraction_model: str = "gpt-5-mini",  # Use fast model for extraction
        output_dir: str = "iteration_logs",
        validation_split: bool = False,  # Enable train/validation/test split
        validation_size: int = 100,  # Size of fixed validation set
        test_size: int = 200,  # Size of held-out test set
        n_train_folds: int = 5,  # Number of different train samples to rotate through
        broad_analysis: bool = True,  # Run one-time broad pattern discovery
        broad_analysis_sample_size: int = 200,  # Number of posts for broad discovery
    ):
        self.client = OpenAI()
        self.data_path = data_path
        self.sample_size = sample_size
        self.max_swaps = max_swaps
        self.analysis_model = analysis_model
        self.extraction_model = extraction_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.validation_split = validation_split
        self.broad_analysis = broad_analysis
        self.broad_analysis_sample_size = broad_analysis_sample_size
        self.broad_patterns: list[str] | None = None
        
        # Load data
        full_df = load_data(data_path)
        
        if validation_split:
            # Split into test (held out), validation (fixed), and train pool
            self._setup_validation_split(full_df, validation_size, test_size, n_train_folds)
        else:
            # Legacy mode: single sample used for all iterations
            self.df = full_df
            if sample_size > 0 and len(self.df) > sample_size:
                self.df = self._stratified_sample(self.df, sample_size)
            self.train_pool = None
            self.validation_df = None
            self.test_df = None
            self.train_folds = None
        
        # Track history
        self.history: list[dict] = []
        self.current_features: dict = copy.deepcopy(DEFAULT_FEATURES)
        self.tried_features: set[str] = set()  # Track features we've tried to avoid re-proposing
        
        # Initialize with current feature names
        for f in self.current_features["features"]:
            self.tried_features.add(f["name"])
        
        # Session ID for this run
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store best model for final evaluation
        self.best_model = None
        self.best_val_accuracy = 0.0
        self.best_features = None
        self.best_feature_columns = None  # Store the column names used in training
        
        # Initialize verbose logger
        self.verbose_logger = VerboseLogger(self.output_dir, self.session_id)
        self._log_initialization()
        
        self._print_init_summary()
    
    def discover_broad_patterns(self) -> list[str]:
        """
        One-time broad pattern discovery (Part 2.1).
        
        Uses a larger sample (100-200) of posts (text + verdict only) to produce
        stable domain priors that ground per-iteration error analysis and feature proposals.
        """
        if self.broad_patterns is not None:
            return self.broad_patterns
        
        if not self.broad_analysis:
            self.broad_patterns = []
            return self.broad_patterns
        
        # Choose source pool for sampling:
        # - validation split: sample from the *train pool* to avoid contaminating test
        # - legacy: sample from the working df
        if self.validation_split and self.train_pool is not None:
            pool = self.train_pool
        else:
            pool = self.df
        
        n = min(self.broad_analysis_sample_size, len(pool))
        # Stratified sample for stability
        sample_df = self._stratified_sample(pool, n, random_state=2026)
        
        examples = []
        for _, row in sample_df.iterrows():
            text = str(row["body"])
            verdict = str(row["verdict"])
            # keep text short to control token usage
            if len(text) > 500:
                text = text[:500] + "..."
            examples.append(f"- VERDICT: {verdict}\n  TEXT: {text}")
        
        prompt = f"""You are analyzing Reddit AITA posts to find broad, general patterns that distinguish verdicts.

IMPORTANT: Verdict labels reflect CROWD CONSENSUS (Reddit commenters), not objective moral truth.
We want patterns that predict crowd perception.

You will see {len(sample_df)} posts. Produce 10-15 patterns that are:
- General across many situations (not "wedding" / "roommate food theft" level specificity)
- Concrete and actionable (phrased as observable signals in the text)
- Focused on crowd perception (framing, tone, norm violations, power, boundary setting, receipts, etc.)

POSTS:
{chr(10).join(examples)}
"""
        response = self.client.responses.parse(
            model=self.analysis_model,
            input=[
                {
                    "role": "system",
                    "content": "You are a careful social-science style analyst. Extract generalizable patterns predicting Reddit crowd consensus. Avoid overfitting to specific topics.",
                },
                {"role": "user", "content": prompt},
            ],
            text_format=BroadPatternDiscovery,
        )
        
        parsed = response.output_parsed
        patterns = parsed.patterns if parsed is not None else []
        # sanitize: keep 15 max to control prompt bloat later
        patterns = [p.strip() for p in patterns if p and p.strip()][:15]
        
        self.broad_patterns = patterns
        self.verbose_logger.log_broad_patterns(patterns, sample_size=len(sample_df))
        return patterns
    
    def _log_initialization(self):
        """Log initialization to verbose log file."""
        self.verbose_logger.log_initialization(
            data_path=self.data_path,
            sample_size=self.sample_size,
            max_swaps=self.max_swaps,
            analysis_model=self.analysis_model,
            extraction_model=self.extraction_model,
            validation_split=self.validation_split,
            train_pool_size=len(self.train_pool) if self.train_pool is not None else None,
            validation_size=len(self.validation_df) if self.validation_df is not None else None,
            test_size=len(self.test_df) if self.test_df is not None else None,
            n_folds=len(self.train_folds) if self.train_folds is not None else None,
            initial_features=[f["name"] for f in self.current_features["features"]],
        )
    
    def _setup_validation_split(
        self, 
        full_df: pd.DataFrame, 
        validation_size: int, 
        test_size: int,
        n_train_folds: int,
    ):
        """Set up train/validation/test splits."""
        # First, separate out test set (never touched until final evaluation)
        remaining_df, self.test_df = self._stratified_split(
            full_df, test_size, random_state=42
        )
        
        # Then separate validation set (fixed across iterations)
        self.train_pool, self.validation_df = self._stratified_split(
            remaining_df, validation_size, random_state=43
        )
        
        # Create multiple train folds for rotation
        self.train_folds = []
        for i in range(n_train_folds):
            fold = self._stratified_sample(
                self.train_pool, 
                self.sample_size, 
                random_state=100 + i
            )
            self.train_folds.append(fold)
        
        # For compatibility, set df to first fold (will be overridden per iteration)
        self.df = self.train_folds[0]
    
    def _stratified_split(
        self, 
        df: pd.DataFrame, 
        split_size: int, 
        random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe into two parts, maintaining label distribution."""
        split_df = self._stratified_sample(df, split_size, random_state=random_state)
        remaining_df = df[~df["id"].isin(split_df["id"])].reset_index(drop=True)
        return remaining_df, split_df
    
    def _stratified_sample(
        self, 
        df: pd.DataFrame, 
        n: int, 
        random_state: int = 42
    ) -> pd.DataFrame:
        """Stratified sample maintaining label distribution."""
        sampled = []
        for label in df["verdict"].unique():
            label_df = df[df["verdict"] == label]
            n_label = max(1, int(n * len(label_df) / len(df)))
            sampled.append(label_df.sample(
                n=min(n_label, len(label_df)), 
                random_state=random_state
            ))
        result = pd.concat(sampled).sample(frac=1, random_state=random_state).reset_index(drop=True)
        return result.head(n)
    
    def _print_init_summary(self):
        """Print initialization summary."""
        print(f"Initialized IterativeImprover")
        print(f"  Data path: {self.data_path}")
        print(f"  Analysis model: {self.analysis_model}")
        print(f"  Extraction model: {self.extraction_model}")
        print(f"  Max swaps per iteration: {self.max_swaps}")
        
        if self.validation_split:
            print(f"  Evaluation strategy: validation split")
            print(f"    Train pool: {len(self.train_pool)} samples")
            print(f"    Train per iteration: {self.sample_size} samples (rotating {len(self.train_folds)} folds)")
            print(f"    Validation: {len(self.validation_df)} samples (fixed)")
            print(f"    Test: {len(self.test_df)} samples (held out)")
        else:
            print(f"  Evaluation strategy: single sample (legacy)")
            print(f"    Sample size: {len(self.df)} samples")
    
    def run_distillation(self, feature_config: dict, df: pd.DataFrame = None) -> dict:
        """Run the distillation pipeline with given features."""
        if df is None:
            df = self.df
            
        distiller = SemanticDistiller(
            feature_config=feature_config,
            model=self.extraction_model,
            use_cache=True,  # Enable caching to avoid re-extracting unchanged features
            max_concurrent=30,
            requests_per_minute=1200,
        )
        
        results = distiller.run(
            df,
            text_col="body",
            label_col="verdict",
            do_residual_analysis=False,  # We'll do our own analysis
        )
        
        return results
    
    def extract_features_only(
        self, 
        feature_config: dict, 
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Extract features without fitting a model. 
        
        Returns (features_df, cost_info).
        """
        distiller = SemanticDistiller(
            feature_config=feature_config,
            model=self.extraction_model,
            use_cache=True,
            max_concurrent=30,
            requests_per_minute=1200,
        )
        
        features_df = distiller.extract_batch(df, text_col="body")
        cost_info = distiller.calculate_cost()
        return features_df, cost_info
    
    def fit_and_evaluate_split(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        val_features: pd.DataFrame,
        val_labels: pd.Series,
    ) -> tuple[dict, Pipeline]:
        """
        Fit model on train set and evaluate on validation set.
        
        Returns (results_dict, fitted_model).
        
        The results_dict includes 'val_preds' (encoded int predictions) and
        'val_pred_probs' (class probabilities) from the actual fitted model,
        so per-class metrics can be computed from the real train->validate
        predictions rather than from a separately-fit model.
        """
        # Build feature matrices
        X_train, feature_names = SemanticDistiller.build_feature_matrix(train_features)
        X_val, _ = SemanticDistiller.build_feature_matrix(val_features)
        
        # Align validation columns to match training columns (see 1.3 fix below)
        missing_in_val = set(X_train.columns) - set(X_val.columns)
        extra_in_val = set(X_val.columns) - set(X_train.columns)
        
        if missing_in_val:
            logging.getLogger("distill").warning(
                f"Validation missing {len(missing_in_val)} columns present in train "
                f"(filling with 0): {sorted(missing_in_val)}"
            )
            for col in missing_in_val:
                X_val[col] = 0
        
        if extra_in_val:
            logging.getLogger("distill").warning(
                f"Validation has {len(extra_in_val)} extra columns not in train "
                f"(dropping): {sorted(extra_in_val)}"
            )
        
        X_val = X_val[X_train.columns]  # Ensure same column order, drop extras
        
        # Sanity assertion: columns must now match exactly
        assert list(X_train.columns) == list(X_val.columns), (
            f"Column mismatch after alignment! "
            f"Train cols: {list(X_train.columns)}, Val cols: {list(X_val.columns)}"
        )
        
        # Align labels with features
        train_labels = train_labels.loc[train_labels.index.astype(str).isin(X_train.index.astype(str))]
        val_labels = val_labels.loc[val_labels.index.astype(str).isin(X_val.index.astype(str))]
        
        # Encode labels
        label_map = {label: i for i, label in enumerate(sorted(train_labels.unique()))}
        reverse_label_map = {v: k for k, v in label_map.items()}
        y_train = train_labels.map(label_map)
        y_val = val_labels.map(label_map)
        
        # Determine safe number of CV splits based on smallest class count
        min_class_count = min(y_train.value_counts())
        safe_inner_cv = max(2, min(3, min_class_count))
        safe_outer_cv = max(2, min(5, min_class_count, len(y_train) // 2))
        
        # Fit model on train
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0],
                cv=safe_inner_cv,
                l1_ratios=(0.5,),
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        pipe.fit(X_train, y_train)
        
        # Evaluate on validation using the ACTUAL fitted model
        val_preds = pipe.predict(X_val)
        val_pred_probs = pipe.predict_proba(X_val)
        val_accuracy = (val_preds == y_val.values).mean()
        
        # Also get CV score on train for comparison
        cv = StratifiedKFold(n_splits=safe_outer_cv, shuffle=True, random_state=42)
        train_cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        
        # Get coefficients
        model = pipe.named_steps["model"]
        coefs = model.coef_[0]
        coef_table = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coef": np.abs(coefs),
            "odds_ratio": np.exp(coefs),
        }).sort_values("abs_coef", ascending=False)
        
        results = {
            "train_cv_accuracy_mean": train_cv_scores.mean(),
            "train_cv_accuracy_std": train_cv_scores.std(),
            "val_accuracy": val_accuracy,
            "n_train": len(y_train),
            "n_val": len(y_val),
            "coefficients": coef_table.to_dict(orient="records"),
            "n_nonzero_features": int((np.abs(coefs) > 1e-6).sum()),
            "regularization_C": float(model.C_[0]),
            "label_map": label_map,
            "reverse_label_map": reverse_label_map,
            "feature_columns": list(X_train.columns),  # Store column names for later use
            # Return actual validation predictions for per-class metrics (fix 1.2)
            "val_preds": val_preds,
            "val_pred_probs": val_pred_probs,
            "y_val": y_val.values,
            "val_index": list(X_val.index),
        }
        
        return results, pipe
    
    def evaluate_on_test(self, model: Pipeline, feature_config: dict) -> dict:
        """Final evaluation on held-out test set."""
        if self.test_df is None:
            raise ValueError("No test set available. Use --validation-split to enable.")
        
        print(f"\n{'='*60}")
        print("FINAL EVALUATION ON HELD-OUT TEST SET")
        print(f"{'='*60}")
        
        # Extract features on test set
        print(f"Extracting features for {len(self.test_df)} test samples...")
        test_features, _ = self.extract_features_only(feature_config, self.test_df)
        
        # Build feature matrix
        X_test, _ = SemanticDistiller.build_feature_matrix(test_features)
        
        # Ensure test has same columns as model was trained on
        if self.best_feature_columns is None:
            raise RuntimeError("best_feature_columns not set. Model must be trained first.")
        model_features = self.best_feature_columns
        missing_cols = set(model_features) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[model_features]
        
        # Get labels
        test_labels = self.test_df.set_index("id")["verdict"]
        test_labels = test_labels.loc[test_labels.index.astype(str).isin(X_test.index.astype(str))]
        
        # Encode and predict
        label_map = {label: i for i, label in enumerate(sorted(test_labels.unique()))}
        y_test = test_labels.map(label_map)
        
        test_preds = model.predict(X_test)
        test_accuracy = (test_preds == y_test.values).mean()
        
        print(f"Test accuracy: {test_accuracy:.3f}")
        print(f"Test samples: {len(y_test)}")
        
        # Compute per-class metrics
        class_names = sorted(test_labels.unique())
        per_class_metrics = compute_per_class_metrics(y_test.values, test_preds, class_names)
        
        # Print per-class metrics to console
        print(f"Per-class metrics:")
        for m in per_class_metrics:
            print(f"  {m['class']}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
        
        # Log test evaluation to verbose log
        self.verbose_logger.log_test_evaluation(
            test_accuracy=test_accuracy,
            n_test=len(y_test),
            best_val_accuracy=self.best_val_accuracy,
            per_class_metrics=per_class_metrics,
        )
        
        return {
            "test_accuracy": test_accuracy,
            "n_test": len(y_test),
            "per_class_metrics": per_class_metrics,
        }
    
    def analyze_samples(
        self,
        predictions_df: pd.DataFrame,
        n_correct: int = 5,
        n_incorrect: int = 5,
    ) -> SampleAnalysis:
        """Use LLM to analyze correctly and incorrectly classified samples."""
        
        # Get samples
        correct = predictions_df[predictions_df["correct"]].sample(
            n=min(n_correct, predictions_df["correct"].sum()),
            random_state=42
        )
        incorrect = predictions_df[~predictions_df["correct"]].sample(
            n=min(n_incorrect, (~predictions_df["correct"]).sum()),
            random_state=42
        )
        
        correct_samples = "\n".join(format_sample_for_prompt(row) for _, row in correct.iterrows())
        incorrect_samples = "\n".join(format_sample_for_prompt(row) for _, row in incorrect.iterrows())
        
        broad_patterns = self.broad_patterns or []
        broad_patterns_text = "\n".join(f"- {p}" for p in broad_patterns) if broad_patterns else "(none)"
        
        prompt = f"""Analyze these classification results from an AITA verdict prediction model.

IMPORTANT CONTEXT: The verdict labels (NTA, YTA, etc.) are NOT objective moral truth.
They represent CROWD CONSENSUS - the weighted aggregate of Reddit commenters' votes.
Our goal is to predict what the crowd will say, not what is objectively "right."
The crowd may have biases, respond to certain writing styles, sympathize with certain
framings, or follow patterns that don't align with an objective moral assessment.

GROUNDING: Broad patterns discovered earlier in this session (use these as priors; do not invent elaborate theories from 10 samples):
{broad_patterns_text}

CORRECTLY CLASSIFIED SAMPLES:
{correct_samples}

INCORRECTLY CLASSIFIED SAMPLES:
{incorrect_samples}

Focused error analysis task:
- Given the broad patterns above, what signals are likely missing or poorly captured by the current feature set?
- What differences distinguish correct from incorrect predictions here?
- Be cautious: you only saw ~10 samples; prefer hypotheses that align with broad patterns.

Consider: How does the poster frame themselves? What narrative techniques do they use?
What might make the crowd sympathetic or unsympathetic, regardless of objective merit?
The current features are extracted values shown above (e.g., self_awareness, empathy_shown, etc.).
"""
        
        response = self.client.responses.parse(
            model=self.analysis_model,
            input=[
                {"role": "system", "content": "You are a machine learning analyst helping improve a text classification system. Remember: you're predicting crowd consensus on Reddit, not objective moral judgment. The crowd has patterns and biases worth modeling. Be specific and actionable."},
                {"role": "user", "content": prompt},
            ],
            text_format=SampleAnalysis,
        )
        
        return response.output_parsed
    
    def evaluate_and_propose_features(
        self,
        results: dict,
        sample_analysis: SampleAnalysis,
        *,
        effective_dimensionality: int | None = None,
        dimensionality_budget: int | None = None,
        top_column_costs: list[tuple[str, int]] | None = None,
    ) -> IterationAnalysis:
        """Use LLM to evaluate current features and propose improvements."""
        
        # Format coefficient info
        coef_info = "\n".join(
            f"  {c['feature']:30s} coef={c['coefficient']:+.4f} (OR={c['odds_ratio']:.3f})"
            for c in sorted(results["results"]["coefficients"], key=lambda x: -abs(x["coefficient"]))
        )
        
        # Current feature descriptions
        feature_info = "\n".join(
            f"  {f['name']}: {f['description']}"
            for f in self.current_features["features"]
        )
        
        broad_patterns = self.broad_patterns or []
        broad_patterns_text = "\n".join(f"- {p}" for p in broad_patterns) if broad_patterns else "(none)"
        
        dim_line = ""
        if effective_dimensionality is not None and dimensionality_budget is not None:
            remaining = dimensionality_budget - effective_dimensionality
            dim_line = (
                f"- Effective dimensionality (post one-hot): {effective_dimensionality} columns\n"
                f"- Dimensionality budget (hard cap): {dimensionality_budget} columns\n"
                f"- Remaining budget: {remaining:+d} columns\n"
            )
        
        cost_lines = ""
        if top_column_costs:
            cost_lines = "\n".join(f"  - {name}: {cost} cols" for name, cost in top_column_costs)
            cost_lines = f"Top per-feature column costs (post-encoding):\n{cost_lines}\n"
        
        prompt = f"""Evaluate this iteration's results and propose feature improvements.

IMPORTANT CONTEXT: The verdict labels (NTA, YTA, etc.) are NOT objective moral truth.
They represent CROWD CONSENSUS - the weighted aggregate of Reddit commenters' votes.
Our goal is to predict what the crowd will say, not what is objectively "right."
The crowd may have biases, respond to certain writing styles, sympathize with certain
framings, or follow patterns that don't align with an objective moral assessment.
Features should capture what ACTUALLY influences crowd perception.

{dim_line}{cost_lines}

Broad domain patterns discovered earlier (ground your proposals in these; avoid overfitting to the 10 sample-analysis posts):
{broad_patterns_text}

CURRENT PERFORMANCE:
- CV Accuracy: {results['results']['cv_accuracy_mean']:.3f} (±{results['results']['cv_accuracy_std']:.3f})
- Non-zero features after L1: {results['results']['n_nonzero_features']}/{len(results['features'])}

FEATURE COEFFICIENTS (sorted by importance):
{coef_info}

CURRENT FEATURE DEFINITIONS:
{feature_info}

SAMPLE ANALYSIS INSIGHTS:
- Correct patterns: {sample_analysis.correct_patterns}
- Incorrect patterns: {sample_analysis.incorrect_patterns}
- Missing signals: {', '.join(sample_analysis.missing_signals)}
- Key insight: {sample_analysis.key_insight}

TASK:
1. Evaluate each feature's usefulness based on its coefficient and the sample analysis
2. Identify up to {self.max_swaps} features to REMOVE (prioritize features with low coefficients and/or high dimensionality cost)
3. Propose up to {self.max_swaps} features to ADD that address the missing signals identified
4. Each replacement feature should be well-defined and extractable from text

Rules for proposed features:
- Use snake_case names
- STRONGLY PREFER binary 'bool' features. Why: we're sample-size constrained, ordinal extraction is noisy, and logistic regression assumes linear effects for scales.
- If you propose a 'scale', prefer 1-3 scales unless there is a clear continuous dimension that benefits from more granularity.
- For 'scale' type: include min_val and max_val (prefer 1-3; use 1-5 only with justification)
- For 'categorical' type: include a values list
- For 'bool' type: just include the description
- Make descriptions specific enough for consistent extraction
- Don't propose features that leak the outcome (verdict)
- Focus on what influences CROWD PERCEPTION, not objective moral assessment. Consider:
  * How does the poster present themselves? (sympathetic vs unsympathetic framing)
  * What narrative techniques might sway crowd opinion?
  * What biases or patterns does the Reddit crowd tend to exhibit?
- AVOID overly specific or sparse features that would only apply to a small subset of samples (e.g., "happened_in_nevada", "involves_wedding"). Prefer features that capture general behavioral/psychological patterns applicable across many situations.
- STRONGLY PREFER bool > scale >>> categorical. We are sample-size constrained and high-cardinality categoricals blow up the feature space (each category becomes a separate one-hot column), reducing statistical power.
  * If you want to capture "is any of these patterns present", use a BOOL, not a categorical listing them all.
  * If you must use categorical, limit to 2-4 broad buckets maximum (e.g., "low/medium/high" or "self/other/mutual").
  * A categorical with 5+ values is almost never appropriate for this problem.
- Respect the dimensionality budget: if we're over budget, prioritize net REMOVALS and dropping expensive categoricals; if under budget, you may add features, but keep within the remaining budget.
- AVOID these already-tried feature names: {', '.join(self.tried_features) if self.tried_features else 'none'}
"""
        
        response = self.client.responses.parse(
            model=self.analysis_model,
            input=[
                {"role": "system", "content": "You are a feature engineering expert. Remember: you're predicting Reddit crowd consensus, not objective moral truth. The crowd has patterns and biases worth modeling. Propose specific, extractable features that will improve prediction of crowd verdicts."},
                {"role": "user", "content": prompt},
            ],
            text_format=IterationAnalysis,
        )
        
        return response.output_parsed
    
    def apply_feature_changes(
        self,
        analysis: IterationAnalysis,
        *,
        dimensionality_budget: int | None = None,
    ) -> dict:
        """Apply the proposed feature changes to create a new feature config."""
        
        new_config = copy.deepcopy(self.current_features)
        
        # Remove features marked for removal
        features_to_remove = set(analysis.features_to_remove[:self.max_swaps])
        new_config["features"] = [
            f for f in new_config["features"]
            if f["name"] not in features_to_remove
        ]
        
        # Add proposed features (variable count; no forced 1:1 swap)
        for proposed in analysis.proposed_features[: self.max_swaps]:
            new_feature = feature_config_to_dict(proposed.model_dump())
            # Avoid accidental duplicates
            if any(f.get("name") == new_feature["name"] for f in new_config["features"]):
                continue
            new_config["features"].append(new_feature)
            # Track this feature so we don't re-propose it
            self.tried_features.add(proposed.name)
        
        # Update the name to reflect iteration
        iteration_num = len(self.history) + 1
        new_config["name"] = f"AITAFeatures_iter{iteration_num}"
        
        # Enforce dimensionality budget for next iteration (Part 2.2 / 2.4)
        budget_info = {}
        if dimensionality_budget is not None:
            new_config, budget_info = enforce_dimensionality_budget_on_config(new_config, dimensionality_budget)
            # If we auto-removed features, also add them to tried_features so we don't re-add immediately
            for fname in budget_info.get("auto_removed_for_budget", []):
                if fname:
                    self.tried_features.add(fname)
        
        # Attach budget info for caller to record/log if desired
        new_config["_budget_info"] = budget_info
        return new_config
    
    def run_iteration(self, iteration_num: int) -> dict:
        """Run a single iteration of the improvement process."""
        
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration_num}")
        print(f"{'='*60}")
        print(f"Current features: {[f['name'] for f in self.current_features['features']]}")
        
        if self.validation_split:
            return self._run_iteration_with_validation(iteration_num)
        else:
            return self._run_iteration_legacy(iteration_num)
    
    def _run_iteration_with_validation(self, iteration_num: int) -> dict:
        """Run iteration with train/validation split (recommended)."""
        
        # Select train fold for this iteration (rotate through folds)
        fold_idx = (iteration_num - 1) % len(self.train_folds)
        train_df = self.train_folds[fold_idx]
        print(f"Using train fold {fold_idx + 1}/{len(self.train_folds)} ({len(train_df)} samples)")
        
        # Log iteration start with current features
        self.verbose_logger.log_iteration_start(iteration_num, self.current_features["features"])
        
        # Track costs across extraction calls
        total_cost = 0.0
        
        # Step 1: Extract features on train set
        print("\n[1/5] Extracting features on train set...")
        train_features, train_cost = self.extract_features_only(self.current_features, train_df)
        total_cost += train_cost.get("total_cost", 0.0)
        print(f"      Extracted {len(train_features)} samples (${train_cost.get('total_cost', 0):.4f})")
        
        # Step 2: Extract features on validation set
        print("\n[2/5] Extracting features on validation set...")
        val_features, val_cost = self.extract_features_only(self.current_features, self.validation_df)
        total_cost += val_cost.get("total_cost", 0.0)
        print(f"      Extracted {len(val_features)} samples (${val_cost.get('total_cost', 0):.4f})")
        
        # Step 3: Fit model on train, evaluate on validation
        print("\n[3/5] Fitting model and evaluating...")
        train_labels = train_df.set_index("id")["verdict"]
        val_labels = self.validation_df.set_index("id")["verdict"]
        
        # Compute effective dimensionality + budget (post one-hot) for this fold
        X_train_tmp, _ = SemanticDistiller.build_feature_matrix(train_features)
        effective_dim = int(X_train_tmp.shape[1])
        dim_budget = compute_dimensionality_budget(n_train=len(train_features))
        costs_top = top_feature_column_costs(X_train_tmp, train_features, top_k=10)
        if effective_dim > dim_budget:
            logging.getLogger("distill").warning(
                f"Effective dimensionality {effective_dim} exceeds budget {dim_budget}. "
                "Expect overfitting; the next iteration should prioritize net removals / cheaper features."
            )
        
        eval_results, fitted_model = self.fit_and_evaluate_split(
            train_features, train_labels,
            val_features, val_labels
        )
        
        print(f"      Train CV Accuracy: {eval_results['train_cv_accuracy_mean']:.3f} (±{eval_results['train_cv_accuracy_std']:.3f})")
        print(f"      Validation Accuracy: {eval_results['val_accuracy']:.3f}")
        
        # Track best model
        is_best = eval_results['val_accuracy'] > self.best_val_accuracy
        if is_best:
            self.best_val_accuracy = eval_results['val_accuracy']
            self.best_model = fitted_model
            self.best_features = copy.deepcopy(self.current_features)
            self.best_feature_columns = eval_results['feature_columns']
            print(f"      ★ New best validation accuracy!")
        
        # Log evaluation results to verbose log
        self.verbose_logger.log_evaluation_results(
            train_cv_accuracy=eval_results['train_cv_accuracy_mean'],
            train_cv_std=eval_results['train_cv_accuracy_std'],
            val_accuracy=eval_results['val_accuracy'],
            n_train=eval_results['n_train'],
            n_val=eval_results['n_val'],
            n_nonzero_features=eval_results['n_nonzero_features'],
            total_features=len(eval_results['coefficients']),
            regularization_c=eval_results.get('regularization_C'),
            is_best=is_best,
        )
        
        # Log feature coefficients
        self.verbose_logger.log_coefficients(eval_results['coefficients'])
        
        # Step 4: Analyze predictions on validation set using ACTUAL fitted model
        # (Fix 1.2: Previously this called get_predictions_with_details() which
        #  fit a SEPARATE model on the validation data and predicted on that same
        #  data. Now we use the actual train->validate predictions from the fitted
        #  model returned by fit_and_evaluate_split().)
        print("\n[4/5] Analyzing validation predictions with LLM...")
        
        # Build predictions DataFrame from actual model outputs
        reverse_label_map = eval_results['reverse_label_map']
        val_preds_encoded = eval_results['val_preds']
        val_pred_probs = eval_results['val_pred_probs']
        val_true_encoded = eval_results['y_val']
        val_idx = eval_results['val_index']
        
        val_texts = self.validation_df.set_index("id")["body"]
        val_texts.index = val_texts.index.astype(str)
        
        val_predictions_df = pd.DataFrame({
            "text": [val_texts.loc[str(idx)] if str(idx) in val_texts.index else "" for idx in val_idx],
            "true_label": [reverse_label_map[t] for t in val_true_encoded],
            "predicted_label": [reverse_label_map[p] for p in val_preds_encoded],
            "correct": val_preds_encoded == val_true_encoded,
            "confidence": val_pred_probs.max(axis=1),
        }, index=[str(idx) for idx in val_idx])
        
        # Add feature values for the sample analysis prompt
        val_features.index = val_features.index.astype(str)
        for col in val_features.columns:
            str_idx = [str(idx) for idx in val_idx]
            val_predictions_df[f"feat_{col}"] = val_features.loc[
                val_features.index.isin(str_idx), col
            ].reindex(str_idx).values
        
        n_correct = val_predictions_df["correct"].sum()
        n_total = len(val_predictions_df)
        print(f"      Validation Correct: {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")
        
        # Compute per-class metrics from the ACTUAL fitted model's predictions
        class_names = sorted(val_labels.unique())
        label_to_idx = {label: i for i, label in enumerate(class_names)}
        per_class_metrics = compute_per_class_metrics(
            val_true_encoded, val_preds_encoded, class_names
        )
        
        # Print per-class metrics to console
        print(f"      Per-class metrics:")
        for m in per_class_metrics:
            print(f"        {m['class']}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
        
        # Log per-class metrics to verbose log
        self.verbose_logger.log_per_class_metrics(per_class_metrics)
        
        sample_analysis = self.analyze_samples(val_predictions_df)
        print(f"      Key insight: {sample_analysis.key_insight}")
        
        # Log sample analysis insights
        self.verbose_logger.log_sample_analysis(
            correct_patterns=sample_analysis.correct_patterns,
            incorrect_patterns=sample_analysis.incorrect_patterns,
            missing_signals=sample_analysis.missing_signals,
            key_insight=sample_analysis.key_insight,
        )
        
        # Step 5: Evaluate features and propose changes
        print("\n[5/5] Evaluating features and proposing changes...")
        
        # Adapt results format for the analysis function
        results_for_analysis = {
            "results": {
                "cv_accuracy_mean": eval_results['val_accuracy'],  # Use val accuracy as primary metric
                "cv_accuracy_std": eval_results['train_cv_accuracy_std'],
                "coefficients": eval_results['coefficients'],
                "n_nonzero_features": eval_results['n_nonzero_features'],
            },
            "features": [c['feature'] for c in eval_results['coefficients']],
        }
        
        iteration_analysis = self.evaluate_and_propose_features(
            results_for_analysis,
            sample_analysis,
            effective_dimensionality=effective_dim,
            dimensionality_budget=dim_budget,
            top_column_costs=costs_top,
        )
        print(f"      Summary: {iteration_analysis.summary}")
        print(f"      Features to remove: {iteration_analysis.features_to_remove}")
        print(f"      Features to add: {[f.name for f in iteration_analysis.proposed_features]}")
        
        # Log feature evaluations with reasoning
        self.verbose_logger.log_feature_evaluations(
            [e.model_dump() for e in iteration_analysis.feature_evaluations]
        )
        
        # Log feature change decisions
        self.verbose_logger.log_feature_changes(
            features_removed=iteration_analysis.features_to_remove,
            features_added=[f.model_dump() for f in iteration_analysis.proposed_features],
            expected_improvement=iteration_analysis.expected_improvement,
        )
        
        # Log iteration summary
        self.verbose_logger.log_iteration_summary(iteration_analysis.summary)
        
        # Apply changes for next iteration
        new_features = self.apply_feature_changes(
            iteration_analysis,
            dimensionality_budget=dim_budget,
        )
        budget_info = new_features.pop("_budget_info", {}) if isinstance(new_features, dict) else {}
        
        # Record history
        iteration_record = {
            "iteration": iteration_num,
            "timestamp": datetime.now().isoformat(),
            "train_fold": fold_idx,
            "features_used": [f["name"] for f in self.current_features["features"]],
            "train_cv_accuracy": eval_results['train_cv_accuracy_mean'],
            "train_cv_std": eval_results['train_cv_accuracy_std'],
            "val_accuracy": eval_results['val_accuracy'],
            "n_train": eval_results['n_train'],
            "n_val": eval_results['n_val'],
            "effective_dimensionality": effective_dim,
            "dimensionality_budget": dim_budget,
            "estimated_config_effective_dim": estimate_config_effective_dim(self.current_features),
            "budget_enforcement": budget_info,
            "per_class_metrics": per_class_metrics,
            "extraction_cost": total_cost,
            "sample_analysis": {
                "correct_patterns": sample_analysis.correct_patterns,
                "incorrect_patterns": sample_analysis.incorrect_patterns,
                "missing_signals": sample_analysis.missing_signals,
                "key_insight": sample_analysis.key_insight,
            },
            "feature_evaluations": [e.model_dump() for e in iteration_analysis.feature_evaluations],
            "features_removed": iteration_analysis.features_to_remove,
            "features_added": [f.model_dump() for f in iteration_analysis.proposed_features],
            "expected_improvement": iteration_analysis.expected_improvement,
        }
        self.history.append(iteration_record)
        
        # Update current features for next iteration
        self.current_features = new_features
        
        # Save iteration results
        self._save_iteration(iteration_record)
        
        return iteration_record
    
    def _run_iteration_legacy(self, iteration_num: int) -> dict:
        """Run iteration with legacy single-sample approach."""
        
        # Log iteration start with current features
        self.verbose_logger.log_iteration_start(iteration_num, self.current_features["features"])
        
        # Step 1: Run distillation
        print("\n[1/4] Running distillation pipeline...")
        results = self.run_distillation(self.current_features)
        accuracy = results["results"]["cv_accuracy_mean"]
        print(f"      CV Accuracy: {accuracy:.3f}")
        
        # Step 2: Get detailed predictions for analysis
        print("\n[2/4] Analyzing predictions...")
        # Features are saved in the logs directory (relative to where distillation runs)
        features_path = Path("logs") / f"features_{results['run_id']}.csv"
        features_df = pd.read_csv(features_path, index_col=0)
        
        predictions_df = get_predictions_with_details(
            features_df,
            self.df.set_index("id")["verdict"],
            self.df.set_index("id")["body"],
        )
        
        n_correct = predictions_df["correct"].sum()
        n_total = len(predictions_df)
        print(f"      Correct: {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")
        
        # Compute and log per-class metrics
        true_labels = predictions_df["true_label"]
        class_names = sorted(true_labels.unique())
        label_to_idx = {label: i for i, label in enumerate(class_names)}
        y_true = predictions_df["true_label"].map(label_to_idx).values
        y_pred = predictions_df["predicted_label"].map(label_to_idx).values
        
        per_class_metrics = compute_per_class_metrics(y_true, y_pred, class_names)
        
        # Print per-class metrics to console
        print(f"      Per-class metrics:")
        for m in per_class_metrics:
            print(f"        {m['class']}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
        
        # Log evaluation results
        self.verbose_logger.log_evaluation_results(
            train_cv_accuracy=results["results"]["cv_accuracy_mean"],
            train_cv_std=results["results"]["cv_accuracy_std"],
            n_train=n_total,
            n_nonzero_features=results["results"]["n_nonzero_features"],
            total_features=len(results["results"]["coefficients"]),
        )
        
        # Log coefficients
        self.verbose_logger.log_coefficients(results["results"]["coefficients"])
        
        # Log per-class metrics
        self.verbose_logger.log_per_class_metrics(per_class_metrics)
        
        # Step 3: LLM analysis of samples
        print("\n[3/4] Analyzing samples with LLM...")
        sample_analysis = self.analyze_samples(predictions_df)
        print(f"      Key insight: {sample_analysis.key_insight}")
        
        # Log sample analysis
        self.verbose_logger.log_sample_analysis(
            correct_patterns=sample_analysis.correct_patterns,
            incorrect_patterns=sample_analysis.incorrect_patterns,
            missing_signals=sample_analysis.missing_signals,
            key_insight=sample_analysis.key_insight,
        )
        
        # Step 4: Evaluate features and propose changes
        print("\n[4/4] Evaluating features and proposing changes...")
        iteration_analysis = self.evaluate_and_propose_features(
            results,
            sample_analysis,
            effective_dimensionality=effective_dim,
            dimensionality_budget=dim_budget,
            top_column_costs=costs_top,
        )
        print(f"      Summary: {iteration_analysis.summary}")
        print(f"      Features to remove: {iteration_analysis.features_to_remove}")
        print(f"      Features to add: {[f.name for f in iteration_analysis.proposed_features]}")
        
        # Log feature evaluations with reasoning
        self.verbose_logger.log_feature_evaluations(
            [e.model_dump() for e in iteration_analysis.feature_evaluations]
        )
        
        # Log feature change decisions
        self.verbose_logger.log_feature_changes(
            features_removed=iteration_analysis.features_to_remove,
            features_added=[f.model_dump() for f in iteration_analysis.proposed_features],
            expected_improvement=iteration_analysis.expected_improvement,
        )
        
        # Log iteration summary
        self.verbose_logger.log_iteration_summary(iteration_analysis.summary)
        
        # Apply changes for next iteration
        # Budget derived from current extracted feature matrix size if available
        effective_dim = None
        dim_budget = None
        costs_top = None
        try:
            X_tmp, _ = SemanticDistiller.build_feature_matrix(features_df)
            effective_dim = int(X_tmp.shape[1])
            dim_budget = compute_dimensionality_budget(n_train=len(features_df))
            costs_top = top_feature_column_costs(X_tmp, features_df, top_k=10)
        except Exception:
            pass
        
        new_features = self.apply_feature_changes(
            iteration_analysis,
            dimensionality_budget=dim_budget,
        )
        budget_info = new_features.pop("_budget_info", {}) if isinstance(new_features, dict) else {}
        
        # Record history
        iteration_record = {
            "iteration": iteration_num,
            "timestamp": datetime.now().isoformat(),
            "features_used": [f["name"] for f in self.current_features["features"]],
            "cv_accuracy": accuracy,
            "cv_std": results["results"]["cv_accuracy_std"],
            "train_accuracy": n_correct / n_total,
            "effective_dimensionality": effective_dim,
            "dimensionality_budget": dim_budget,
            "estimated_config_effective_dim": estimate_config_effective_dim(self.current_features),
            "budget_enforcement": budget_info,
            "per_class_metrics": per_class_metrics,
            "sample_analysis": {
                "correct_patterns": sample_analysis.correct_patterns,
                "incorrect_patterns": sample_analysis.incorrect_patterns,
                "missing_signals": sample_analysis.missing_signals,
                "key_insight": sample_analysis.key_insight,
            },
            "feature_evaluations": [e.model_dump() for e in iteration_analysis.feature_evaluations],
            "features_removed": iteration_analysis.features_to_remove,
            "features_added": [f.model_dump() for f in iteration_analysis.proposed_features],
            "expected_improvement": iteration_analysis.expected_improvement,
            "cost": results["cost"],
            "run_id": results["run_id"],
        }
        self.history.append(iteration_record)
        
        # Update current features for next iteration
        self.current_features = new_features
        
        # Save iteration results
        self._save_iteration(iteration_record)
        
        return iteration_record
    
    def _save_iteration(self, record: dict):
        """Save iteration results to disk."""
        # Save individual iteration
        iter_path = self.output_dir / f"iteration_{record['iteration']:02d}_{self.session_id}.json"
        iter_path.write_text(json.dumps(record, indent=2, default=str))
        
        # Save current feature config
        config_path = self.output_dir / f"features_iter{record['iteration']:02d}_{self.session_id}.json"
        config_path.write_text(json.dumps(self.current_features, indent=2))
        
        # Save full history
        history_path = self.output_dir / f"history_{self.session_id}.json"
        history_path.write_text(json.dumps(self.history, indent=2, default=str))
    
    def run(self, n_iterations: int) -> list[dict]:
        """Run multiple iterations of improvement."""
        
        print(f"\n{'#'*60}")
        print(f"# ITERATIVE FEATURE IMPROVEMENT")
        print(f"# Session: {self.session_id}")
        print(f"# Planned iterations: {n_iterations}")
        if self.validation_split:
            print(f"# Evaluation strategy: train/validation/test split")
        else:
            print(f"# Evaluation strategy: single sample (legacy)")
        print(f"{'#'*60}")
        
        # Part 2.1: One-time broad pattern discovery (pre-loop)
        if self.broad_analysis:
            print("\nRunning broad pattern discovery (pre-loop)...")
            try:
                patterns = self.discover_broad_patterns()
                print(f"Discovered {len(patterns)} broad patterns.")
            except Exception as e:
                print(f"Broad pattern discovery failed (continuing without it): {e}")
        
        for i in range(1, n_iterations + 1):
            try:
                self.run_iteration(i)
            except Exception as e:
                print(f"\nError in iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Final summary
        self._print_summary()
        
        # Final evaluation on test set (only in validation split mode)
        if self.validation_split and self.best_model is not None:
            test_results = self.evaluate_on_test(self.best_model, self.best_features)
            
            # Save test results
            test_path = self.output_dir / f"test_results_{self.session_id}.json"
            test_path.write_text(json.dumps({
                "test_accuracy": test_results["test_accuracy"],
                "n_test": test_results["n_test"],
                "best_val_accuracy": self.best_val_accuracy,
                "best_features": self.best_features,
            }, indent=2))
            print(f"\nTest results saved to: {test_path}")
        
        return self.history
    
    def _print_summary(self):
        """Print a summary of all iterations."""
        
        print(f"\n{'='*60}")
        print("IMPROVEMENT SUMMARY")
        print(f"{'='*60}")
        
        if not self.history:
            print("No iterations completed.")
            return
        
        if self.validation_split:
            self._print_summary_validation_split()
        else:
            self._print_summary_legacy()
        
        # Log final summary to verbose log
        if self.history:
            if self.validation_split:
                best = max(self.history, key=lambda x: x["val_accuracy"])
                best_accuracy = best["val_accuracy"]
            else:
                best = max(self.history, key=lambda x: x["cv_accuracy"])
                best_accuracy = best["cv_accuracy"]
            
            self.verbose_logger.log_final_summary(
                history=self.history,
                best_iteration=best["iteration"],
                best_accuracy=best_accuracy,
                best_features=best["features_used"],
                validation_split=self.validation_split,
            )
            print(f"\nVerbose log saved to: {self.verbose_logger.log_path}")
    
    def _print_summary_validation_split(self):
        """Print summary for validation split mode."""
        print(f"\n{'Iter':<6} {'Fold':<6} {'Train CV':<12} {'Val Acc':<12} {'Features Changed'}")
        print("-" * 70)
        
        for record in self.history:
            removed = ", ".join(record["features_removed"][:2])
            if len(record["features_removed"]) > 2:
                removed += f" +{len(record['features_removed'])-2}"
            added = ", ".join(f["name"] for f in record["features_added"][:2])
            if len(record["features_added"]) > 2:
                added += f" +{len(record['features_added'])-2}"
            changes = f"-({removed}) +({added})" if removed else "baseline"
            
            print(f"{record['iteration']:<6} {record['train_fold']+1:<6} "
                  f"{record['train_cv_accuracy']:.3f}±{record['train_cv_std']:.2f}  "
                  f"{record['val_accuracy']:.3f}        {changes}")
        
        # Improvement (based on validation accuracy)
        if len(self.history) > 1:
            first_acc = self.history[0]["val_accuracy"]
            last_acc = self.history[-1]["val_accuracy"]
            improvement = last_acc - first_acc
            print(f"\nValidation improvement: {improvement:+.3f} ({first_acc:.3f} → {last_acc:.3f})")
        
        # Best iteration (based on validation)
        best = max(self.history, key=lambda x: x["val_accuracy"])
        print(f"\nBest iteration: {best['iteration']} with validation accuracy {best['val_accuracy']:.3f}")
        print(f"Best features: {best['features_used']}")
        
        # Save best config
        if self.best_features:
            final_path = self.output_dir / f"best_features_{self.session_id}.json"
            final_path.write_text(json.dumps(self.best_features, indent=2))
            print(f"\nBest feature config saved to: {final_path}")
    
    def _print_summary_legacy(self):
        """Print summary for legacy single-sample mode."""
        print(f"\n{'Iter':<6} {'CV Accuracy':<15} {'Train Acc':<12} {'Features Changed'}")
        print("-" * 60)
        
        for record in self.history:
            removed = ", ".join(record["features_removed"][:2])
            if len(record["features_removed"]) > 2:
                removed += f" +{len(record['features_removed'])-2}"
            added = ", ".join(f["name"] for f in record["features_added"][:2])
            if len(record["features_added"]) > 2:
                added += f" +{len(record['features_added'])-2}"
            changes = f"-({removed}) +({added})" if removed else "baseline"
            
            print(f"{record['iteration']:<6} {record['cv_accuracy']:.3f} ±{record['cv_std']:.3f}    "
                  f"{record['train_accuracy']:.3f}        {changes}")
        
        # Improvement
        if len(self.history) > 1:
            first_acc = self.history[0]["cv_accuracy"]
            last_acc = self.history[-1]["cv_accuracy"]
            improvement = last_acc - first_acc
            print(f"\nNet improvement: {improvement:+.3f} ({first_acc:.3f} → {last_acc:.3f})")
        
        # Total cost
        if "cost" in self.history[0]:
            total_cost = sum(r["cost"]["total_cost_usd"] for r in self.history)
            print(f"Total API cost: ${total_cost:.4f}")
        
        # Best iteration
        best = max(self.history, key=lambda x: x["cv_accuracy"])
        print(f"\nBest iteration: {best['iteration']} with CV accuracy {best['cv_accuracy']:.3f}")
        print(f"Best features: {best['features_used']}")
        
        # Save final config
        final_path = self.output_dir / f"best_features_{self.session_id}.json"
        best_config = copy.deepcopy(DEFAULT_FEATURES)
        # Find the features config from the best iteration (it's the input, not output)
        if best["iteration"] == len(self.history):
            # Best is current
            best_config = self.current_features
        else:
            # Load from saved file
            config_path = self.output_dir / f"features_iter{best['iteration']:02d}_{self.session_id}.json"
            if config_path.exists():
                best_config = json.loads(config_path.read_text())
        
        final_path.write_text(json.dumps(best_config, indent=2))
        print(f"\nBest feature config saved to: {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Iteratively improve semantic distillation features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (legacy mode - same sample for all iterations)
    python iterative_improvement.py --iterations 5

    # With validation split (recommended for robust feature selection)
    python iterative_improvement.py --iterations 5 --validation-split
    
    # Customized validation split
    python iterative_improvement.py --iterations 10 --validation-split \\
        --sample-size 100 --val-size 150 --test-size 300 --n-folds 10
"""
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=5,
        help="Number of improvement iterations (default: 5)"
    )
    parser.add_argument(
        "--sample-size", "-s", type=int, default=50,
        help="Number of train samples per iteration (default: 50)"
    )
    parser.add_argument(
        "--max-swaps", type=int, default=3,
        help="Max features to swap per iteration (default: 3)"
    )
    parser.add_argument(
        "--data", type=str, default="dataset/aita_dataset.csv",
        help="Path to data file"
    )
    parser.add_argument(
        "--analysis-model", type=str, default="gpt-5",
        help="Model for analysis (default: gpt-5)"
    )
    parser.add_argument(
        "--extraction-model", type=str, default="gpt-5-mini",
        help="Model for feature extraction (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="iteration_logs",
        help="Directory for output files"
    )
    
    # Validation split arguments
    parser.add_argument(
        "--validation-split", "-v", action="store_true",
        help="Enable train/validation/test split for robust evaluation. "
             "Recommended for avoiding overfitting to a single sample."
    )
    parser.add_argument(
        "--val-size", type=int, default=100,
        help="Size of fixed validation set (default: 100). Only used with --validation-split"
    )
    parser.add_argument(
        "--test-size", type=int, default=200,
        help="Size of held-out test set (default: 200). Only used with --validation-split"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of train folds to rotate through (default: 5). Only used with --validation-split"
    )
    
    parser.add_argument(
        "--no-broad-analysis", action="store_true",
        help="Disable one-time broad pattern discovery (Part 2.1)."
    )
    parser.add_argument(
        "--broad-analysis-sample-size", type=int, default=200,
        help="Number of posts for broad pattern discovery (default: 200)."
    )
    
    args = parser.parse_args()
    
    improver = IterativeImprover(
        data_path=args.data,
        sample_size=args.sample_size,
        max_swaps=args.max_swaps,
        analysis_model=args.analysis_model,
        extraction_model=args.extraction_model,
        output_dir=args.output_dir,
        validation_split=args.validation_split,
        validation_size=args.val_size,
        test_size=args.test_size,
        n_train_folds=args.n_folds,
        broad_analysis=not args.no_broad_analysis,
        broad_analysis_sample_size=args.broad_analysis_sample_size,
    )
    
    improver.run(args.iterations)


if __name__ == "__main__":
    main()
