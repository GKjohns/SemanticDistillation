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
"""

import json
import copy
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    proposed_features: list[ProposedFeature] = Field(..., description="New features to add (same count as removed)")
    expected_improvement: str = Field(..., description="What improvement we expect from these changes")


class SampleAnalysis(BaseModel):
    """Analysis of correctly and incorrectly classified samples."""
    correct_patterns: str = Field(..., description="What patterns exist in correctly classified samples")
    incorrect_patterns: str = Field(..., description="What patterns exist in misclassified samples")
    missing_signals: list[str] = Field(..., description="Signals that could help but aren't captured")
    key_insight: str = Field(..., description="The most important insight for improvement")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

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
    
    # Fit model and get predictions
    pipe = Pipeline([
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
        analysis_model: str = "gpt-5",  # Use powerful model for strategic decisions
        extraction_model: str = "gpt-5-mini",  # Use fast model for extraction
        output_dir: str = "iteration_logs",
    ):
        self.client = OpenAI()
        self.data_path = data_path
        self.sample_size = sample_size
        self.max_swaps = max_swaps
        self.analysis_model = analysis_model
        self.extraction_model = extraction_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data once
        self.df = load_data(data_path)
        if sample_size > 0 and len(self.df) > sample_size:
            self.df = self._stratified_sample(self.df, sample_size)
        
        # Track history
        self.history: list[dict] = []
        self.current_features: dict = copy.deepcopy(DEFAULT_FEATURES)
        self.tried_features: set[str] = set()  # Track features we've tried to avoid re-proposing
        
        # Initialize with current feature names
        for f in self.current_features["features"]:
            self.tried_features.add(f["name"])
        
        # Session ID for this run
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Initialized IterativeImprover")
        print(f"  Data: {len(self.df)} samples from {data_path}")
        print(f"  Analysis model: {analysis_model}")
        print(f"  Extraction model: {extraction_model}")
        print(f"  Max swaps per iteration: {max_swaps}")
    
    def _stratified_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Stratified sample maintaining label distribution."""
        sampled = []
        for label in df["verdict"].unique():
            label_df = df[df["verdict"] == label]
            n_label = max(1, int(n * len(label_df) / len(df)))
            sampled.append(label_df.sample(n=min(n_label, len(label_df)), random_state=42))
        result = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
        return result.head(n)
    
    def run_distillation(self, feature_config: dict) -> dict:
        """Run the distillation pipeline with given features."""
        distiller = SemanticDistiller(
            feature_config=feature_config,
            model=self.extraction_model,
            use_cache=True,  # Enable caching to avoid re-extracting unchanged features
            max_concurrent=20,
            requests_per_minute=500,
        )
        
        results = distiller.run(
            self.df,
            text_col="body",
            label_col="verdict",
            do_residual_analysis=False,  # We'll do our own analysis
        )
        
        return results
    
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
        
        prompt = f"""Analyze these classification results from an AITA verdict prediction model.

CORRECTLY CLASSIFIED SAMPLES:
{correct_samples}

INCORRECTLY CLASSIFIED SAMPLES:
{incorrect_samples}

Analyze what patterns distinguish correct from incorrect predictions.
Focus on what signals might be missing from the feature set that could help.
The current features are extracted values shown above (e.g., self_awareness, empathy_shown, etc.).
"""
        
        response = self.client.responses.parse(
            model=self.analysis_model,
            input=[
                {"role": "system", "content": "You are a machine learning analyst helping improve a text classification system. Be specific and actionable."},
                {"role": "user", "content": prompt},
            ],
            text_format=SampleAnalysis,
        )
        
        return response.output_parsed
    
    def evaluate_and_propose_features(
        self,
        results: dict,
        sample_analysis: SampleAnalysis,
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
        
        prompt = f"""Evaluate this iteration's results and propose feature improvements.

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
2. Identify up to {self.max_swaps} features to swap out (prioritize features with low coefficients that aren't capturing useful signal)
3. Propose replacement features that address the missing signals identified
4. Each replacement feature should be well-defined and extractable from text

Rules for proposed features:
- Use snake_case names
- For 'scale' type: include min_val (usually 1) and max_val (usually 5)
- For 'categorical' type: include a values list
- For 'bool' type: just include the description
- Make descriptions specific enough for consistent extraction
- Don't propose features that leak the outcome (verdict)
- AVOID these already-tried feature names: {', '.join(self.tried_features) if self.tried_features else 'none'}
"""
        
        response = self.client.responses.parse(
            model=self.analysis_model,
            input=[
                {"role": "system", "content": "You are a feature engineering expert. Propose specific, extractable features that will improve classification accuracy."},
                {"role": "user", "content": prompt},
            ],
            text_format=IterationAnalysis,
        )
        
        return response.output_parsed
    
    def apply_feature_changes(
        self,
        analysis: IterationAnalysis,
    ) -> dict:
        """Apply the proposed feature changes to create a new feature config."""
        
        new_config = copy.deepcopy(self.current_features)
        
        # Remove features marked for removal
        features_to_remove = set(analysis.features_to_remove[:self.max_swaps])
        new_config["features"] = [
            f for f in new_config["features"]
            if f["name"] not in features_to_remove
        ]
        
        # Add proposed features
        for proposed in analysis.proposed_features[:len(features_to_remove)]:
            new_feature = feature_config_to_dict(proposed.model_dump())
            new_config["features"].append(new_feature)
            # Track this feature so we don't re-propose it
            self.tried_features.add(proposed.name)
        
        # Update the name to reflect iteration
        iteration_num = len(self.history) + 1
        new_config["name"] = f"AITAFeatures_iter{iteration_num}"
        
        return new_config
    
    def run_iteration(self, iteration_num: int) -> dict:
        """Run a single iteration of the improvement process."""
        
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration_num}")
        print(f"{'='*60}")
        print(f"Current features: {[f['name'] for f in self.current_features['features']]}")
        
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
        
        # Step 3: LLM analysis of samples
        print("\n[3/4] Analyzing samples with LLM...")
        sample_analysis = self.analyze_samples(predictions_df)
        print(f"      Key insight: {sample_analysis.key_insight}")
        
        # Step 4: Evaluate features and propose changes
        print("\n[4/4] Evaluating features and proposing changes...")
        iteration_analysis = self.evaluate_and_propose_features(results, sample_analysis)
        print(f"      Summary: {iteration_analysis.summary}")
        print(f"      Features to remove: {iteration_analysis.features_to_remove}")
        print(f"      Features to add: {[f.name for f in iteration_analysis.proposed_features]}")
        
        # Apply changes for next iteration
        new_features = self.apply_feature_changes(iteration_analysis)
        
        # Record history
        iteration_record = {
            "iteration": iteration_num,
            "timestamp": datetime.now().isoformat(),
            "features_used": [f["name"] for f in self.current_features["features"]],
            "cv_accuracy": accuracy,
            "cv_std": results["results"]["cv_accuracy_std"],
            "train_accuracy": n_correct / n_total,
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
        print(f"{'#'*60}")
        
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
        
        return self.history
    
    def _print_summary(self):
        """Print a summary of all iterations."""
        
        print(f"\n{'='*60}")
        print("IMPROVEMENT SUMMARY")
        print(f"{'='*60}")
        
        if not self.history:
            print("No iterations completed.")
            return
        
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
        description="Iteratively improve semantic distillation features"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=5,
        help="Number of improvement iterations (default: 5)"
    )
    parser.add_argument(
        "--sample-size", "-s", type=int, default=50,
        help="Number of samples to use (default: 50)"
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
    
    args = parser.parse_args()
    
    improver = IterativeImprover(
        data_path=args.data,
        sample_size=args.sample_size,
        max_swaps=args.max_swaps,
        analysis_model=args.analysis_model,
        extraction_model=args.extraction_model,
        output_dir=args.output_dir,
    )
    
    improver.run(args.iterations)


if __name__ == "__main__":
    main()
