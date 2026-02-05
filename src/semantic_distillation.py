"""
Semantic Distillation: LLM-Powered Interpretable Feature Engineering

The idea: Take unstructured text (X) and a label (Y), use an LLM to extract
structured, human-interpretable features from X, then fit classical statistical
models on those features. You get the semantic understanding of LLMs with the
interpretability of logistic regression.

This demo uses Reddit's "Am I the Asshole?" (AITA) dataset.
X = the post text describing a conflict
Y = the crowd verdict (YTA / NTA)

Prior art: FELIX (Malberg et al. 2024), CAAFE (Hollmann et al. 2024)
This implementation adds: iterative residual-driven feature search, full
logging pipeline, and a practical demo of the concept.

Usage:
    export OPENAI_API_KEY=sk-...
    python semantic_distillation.py

    # Or with a real CSV:
    python semantic_distillation.py --data aita_posts.csv --text-col body --label-col verdict
    
    # With a custom feature config:
    python semantic_distillation.py --features my_features.json
    
    # Use cached extractions (off by default):
    python semantic_distillation.py --use-cache
    
    # Clear cache before running:
    python semantic_distillation.py --clear-cache
    
    # Parallel extraction with rate limiting:
    python semantic_distillation.py --max-concurrent 5 --rpm 100
"""

import time
import argparse
import threading
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI

# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Local imports
from features import DEFAULT_FEATURES, FeatureSet
from schemas import build_pydantic_model, ResidualAnalysis
from utils import setup_logging, FeatureCache, RateLimiter, save_results, save_features_csv
from data import load_data


# ---------------------------------------------------------------------------
# Global logging setup (will be replaced by SemanticDistiller's logger)
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Model pricing (per million tokens, as of Feb 2026)
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    # GPT-5 models
    "gpt-5-mini": {
        "input": 0.25,    # $0.25 per 1M input tokens
        "output": 2.00,   # $2.00 per 1M output tokens
    },
    "gpt-5": {
        "input": 1.25,    # $1.25 per 1M input tokens
        "output": 10.00,  # $10.00 per 1M output tokens
    },
    "gpt-5.2": {
        "input": 1.75,    # $1.75 per 1M input tokens
        "output": 14.00,  # $14.00 per 1M output tokens
    },
    "gpt-5.2-pro": {
        "input": 21.00,   # $21.00 per 1M input tokens
        "output": 168.00, # $168.00 per 1M output tokens
    },
    # GPT-4 models (legacy)
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
}


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class SemanticDistiller:
    """Extract structured features from text using LLM, fit interpretable models."""

    def __init__(
        self,
        feature_config: dict[str, Any] | None = None,
        model: str = "gpt-5-mini",
        cache_dir: str = "cache",
        log_dir: str = "logs",
        max_retries: int = 2,
        use_cache: bool = False,
        max_concurrent: int = 5,
        requests_per_minute: float = 60.0,
    ):
        """
        Initialize the SemanticDistiller.
        
        Args:
            feature_config: Feature set configuration dict. If None, uses DEFAULT_FEATURES.
                           Can be a dict from features.py or loaded from JSON.
            model: OpenAI model to use for extraction.
            cache_dir: Directory for caching extracted features.
            log_dir: Directory for storing logs and results.
            max_retries: Number of retries for failed extractions.
            use_cache: Whether to use cached extractions. Defaults to False.
            max_concurrent: Maximum concurrent extraction requests. Defaults to 5.
            requests_per_minute: Rate limit for API requests. Defaults to 60 (1/sec).
        """
        self.client = OpenAI()
        self.model = model
        self.cache = FeatureCache(cache_dir)
        self.use_cache = use_cache
        self.log_dir = Path(log_dir)
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.extraction_log: list[dict] = []
        
        # Set up feature configuration
        self.feature_config = feature_config if feature_config is not None else DEFAULT_FEATURES
        self.feature_set = FeatureSet(self.feature_config)
        
        # Build the Pydantic model for structured extraction
        self.FeatureModel = build_pydantic_model(self.feature_config)
        
        # Set up logging
        self.log, self.run_id, self.log_file = setup_logging(log_dir)
        
        # Token tracking (thread-safe)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._token_lock = threading.Lock()
        
        self.log.info(f"Initialized SemanticDistiller with model={model}, use_cache={use_cache}")
        self.log.info(f"Parallelism: max_concurrent={max_concurrent}, rpm={requests_per_minute}")
        self.log.info(f"Feature set: {self.feature_set.name} ({len(self.feature_set.features)} features)")

    # -- Cost estimation ---------------------------------------------------

    def calculate_cost(self) -> dict:
        """Calculate total cost based on token usage."""
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["gpt-5-mini"])
        
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
            "pricing_per_million": pricing,
        }

    # -- Feature extraction ------------------------------------------------

    def extract_features(self, text: str, post_id: str = "") -> Optional[dict]:
        """Extract features from a single post using structured output."""

        # Check cache if enabled
        if self.use_cache:
            cached = self.cache.get(text)
            if cached is not None:
                self.log.info(f"  [{post_id}] Cache hit")
                return cached

        system_prompt = f"""You are a careful text analyst. Given a Reddit AITA post, 
extract the requested features from the text. Be objective and analytical.

IMPORTANT: You are extracting properties of the TEXT ITSELF, not predicting 
the verdict. Focus on what is written, how it's written, and what situation 
is described. Do not consider what verdict the crowd might give.

Feature set: {self.feature_set.name}
{self.feature_set.description}"""

        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                self.rate_limiter.acquire()
                
                t0 = time.time()

                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze this AITA post:\n\n{text}"},
                    ],
                    text_format=self.FeatureModel,
                )

                elapsed = time.time() - t0
                features = response.output_parsed

                if features is None:
                    self.log.warning(f"  [{post_id}] Model refused to extract (attempt {attempt+1})")
                    continue

                result = features.model_dump()

                # Track tokens (thread-safe)
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                with self._token_lock:
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens

                # Log the extraction
                entry = {
                    "post_id": post_id,
                    "model": self.model,
                    "elapsed_s": round(elapsed, 2),
                    "attempt": attempt + 1,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                }
                self.extraction_log.append(entry)
                self.log.info(
                    f"  [{post_id}] Extracted in {elapsed:.1f}s "
                    f"({input_tokens}+{output_tokens} tokens)"
                )

                # Save to cache if enabled
                if self.use_cache:
                    self.cache.set(text, result)
                return result

            except Exception as e:
                self.log.error(f"  [{post_id}] Extraction failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        self.log.error(f"  [{post_id}] All extraction attempts failed")
        return None

    def _extract_single(self, text: str, post_id: str) -> tuple[str, Optional[dict]]:
        """Helper for parallel extraction. Returns (post_id, features)."""
        features = self.extract_features(text, post_id=post_id)
        return post_id, features

    def extract_batch(self, df: pd.DataFrame, text_col: str = "body") -> pd.DataFrame:
        """Extract features for every row in parallel. Returns a features DataFrame."""
        self.log.info(f"Extracting features from {len(df)} posts (max_concurrent={self.max_concurrent})...")

        # Prepare tasks
        tasks = []
        for idx, row in df.iterrows():
            post_id = str(row.get("id", f"row_{idx}"))
            text = row[text_col]
            tasks.append((text, post_id))

        records = []
        failed = []
        
        # Execute in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._extract_single, text, post_id): post_id
                for text, post_id in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                post_id = futures[future]
                try:
                    _, features = future.result()
                    if features is not None:
                        features["_id"] = post_id
                        records.append(features)
                    else:
                        failed.append(post_id)
                except Exception as e:
                    self.log.error(f"  [{post_id}] Unexpected error: {e}")
                    failed.append(post_id)
        
        if failed:
            self.log.warning(f"  Skipped {len(failed)} posts (extraction failed): {failed}")

        features_df = pd.DataFrame(records).set_index("_id")
        self.log.info(f"Successfully extracted features for {len(features_df)}/{len(df)} posts")
        return features_df

    # -- Feature matrix construction ---------------------------------------

    @staticmethod
    def build_feature_matrix(features_df: pd.DataFrame, log=None) -> tuple[pd.DataFrame, list[str]]:
        """Convert extracted features into a numeric matrix for modeling.
        Returns (X_df, feature_names)."""

        X = features_df.copy()

        # One-hot encode categoricals
        categoricals = X.select_dtypes(include=["object"]).columns.tolist()
        if categoricals:
            X = pd.get_dummies(X, columns=categoricals, drop_first=True, dtype=int)

        # Convert booleans to int
        for col in X.select_dtypes(include=["bool"]).columns:
            X[col] = X[col].astype(int)

        feature_names = X.columns.tolist()
        if log:
            log.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
            log.info(f"Features: {feature_names}")

        return X, feature_names

    # -- Modeling -----------------------------------------------------------

    @staticmethod
    def fit_and_evaluate(
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        n_folds: int = 5,
        log=None,
    ) -> dict:
        """Fit logistic regression with CV, return results + coefficients."""

        if log:
            log.info(f"\n{'='*60}")
            log.info("MODELING")
            log.info(f"{'='*60}")
            log.info(f"Samples: {len(y)}, Features: {len(feature_names)}")
            log.info(f"Class balance: {y.value_counts().to_dict()}")

        # Encode labels
        label_map = {label: i for i, label in enumerate(sorted(y.unique()))}
        y_enc = y.map(label_map)
        if log:
            log.info(f"Label mapping: {label_map}")

        # Cross-validation
        cv = StratifiedKFold(n_splits=min(n_folds, len(y) // 2), shuffle=True, random_state=42)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0],  # Less aggressive regularization range
                cv=3,
                l1_ratios=(0.5,),  # Elastic net (mix of L1 and L2)
                solver="saga",
                max_iter=5000,
                class_weight="balanced",  # Handle class imbalance
                random_state=42,
            )),
        ])

        # Score with outer CV
        scores = cross_val_score(pipe, X, y_enc, cv=cv, scoring="accuracy")
        if log:
            log.info(f"\nCross-validation accuracy: {scores.mean():.3f} (±{scores.std():.3f})")
            log.info(f"Fold scores: {[f'{s:.3f}' for s in scores]}")

        # Fit on full data to get coefficients
        pipe.fit(X, y_enc)
        model = pipe.named_steps["model"]
        coefs = model.coef_[0]

        # Build coefficient table
        coef_table = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coef": np.abs(coefs),
            "odds_ratio": np.exp(coefs),
        }).sort_values("abs_coef", ascending=False)

        if log:
            log.info(f"\n{'='*60}")
            log.info("COEFFICIENT TABLE (sorted by importance)")
            log.info(f"{'='*60}")
            log.info(f"Target: {[k for k,v in label_map.items() if v==1][0] if len(label_map)==2 else label_map}")
            log.info(f"Regularization C: {model.C_[0]:.4f}")
            log.info(f"")

            for _, row in coef_table.iterrows():
                direction = "+" if row["coefficient"] > 0 else "-" if row["coefficient"] < 0 else " "
                bar = "█" * int(row["abs_coef"] * 10)
                log.info(
                    f"  {direction} {row['feature']:40s} "
                    f"coef={row['coefficient']:+.4f}  "
                    f"OR={row['odds_ratio']:.3f}  "
                    f"{bar}"
                )

        # Non-zero features (survived L1)
        n_nonzero = (np.abs(coefs) > 1e-6).sum()
        if log:
            log.info(f"\nFeatures surviving L1 regularization: {n_nonzero}/{len(feature_names)}")

        return {
            "cv_accuracy_mean": scores.mean(),
            "cv_accuracy_std": scores.std(),
            "fold_scores": scores.tolist(),
            "coefficients": coef_table.to_dict(orient="records"),
            "n_nonzero_features": int(n_nonzero),
            "regularization_C": float(model.C_[0]),
            "label_map": label_map,
        }

    # -- Residual analysis -------------------------------------------------

    def analyze_residuals(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        texts: pd.Series,
        feature_names: list[str],
        n_examples: int = 5,
    ) -> Optional[ResidualAnalysis]:
        """Find misclassified examples and ask LLM to suggest new features."""

        self.log.info(f"\n{'='*60}")
        self.log.info("RESIDUAL ANALYSIS")
        self.log.info(f"{'='*60}")

        label_map = {label: i for i, label in enumerate(sorted(y.unique()))}
        y_enc = y.map(label_map)

        # Fit model
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                l1_ratio=0.5, solver="saga", C=1.0, max_iter=5000, 
                class_weight="balanced", random_state=42
            )),
        ])
        pipe.fit(X, y_enc)
        preds = pipe.predict(X)

        # Find misclassified
        misclassified_mask = preds != y_enc.values
        n_wrong = misclassified_mask.sum()
        self.log.info(f"Misclassified on training data: {n_wrong}/{len(y)} ({n_wrong/len(y)*100:.1f}%)")

        if n_wrong == 0:
            self.log.info("No misclassifications to analyze!")
            return None

        # Gather misclassified examples
        reverse_map = {v: k for k, v in label_map.items()}
        wrong_indices = np.where(misclassified_mask)[0][:n_examples]
        examples = []
        for idx in wrong_indices:
            examples.append(
                f"---\n"
                f"TRUE LABEL: {reverse_map[y_enc.iloc[idx]]}\n"
                f"PREDICTED: {reverse_map[preds[idx]]}\n"
                f"TEXT: {texts.iloc[idx][:500]}\n"
            )

        examples_text = "\n".join(examples)

        prompt = f"""I'm building a model to predict AITA verdicts from extracted text features.
The following posts were MISCLASSIFIED by my current model.

Current features: {feature_names}

Misclassified examples:
{examples_text}

Analyze what patterns exist in these misclassifications. What aspects of these
texts are NOT captured by the current feature set? Suggest 3-5 new features
that could help the model distinguish these cases correctly.

Each new feature should be:
- Extractable purely from the text (no outcome leakage)
- Distinct from existing features
- Specific and well-defined enough to extract reliably"""

        try:
            t0 = time.time()
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are a data scientist analyzing model failures to improve feature engineering."},
                    {"role": "user", "content": prompt},
                ],
                text_format=ResidualAnalysis,
            )
            elapsed = time.time() - t0

            analysis = response.output_parsed
            if analysis is None:
                self.log.warning("Model refused residual analysis")
                return None

            self.log.info(f"Residual analysis completed in {elapsed:.1f}s")
            self.log.info(f"\nPattern summary: {analysis.pattern_summary}")
            self.log.info(f"\nProposed new features:")
            for h in analysis.hypotheses:
                self.log.info(f"  • {h.feature_name} ({h.feature_type}): {h.description}")

            return analysis

        except Exception as e:
            self.log.error(f"Residual analysis failed: {e}")
            return None

    # -- Full pipeline -----------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        text_col: str = "body",
        label_col: str = "verdict",
        do_residual_analysis: bool = True,
    ) -> dict:
        """Run the full semantic distillation pipeline."""

        self.log.info(f"\n{'#'*60}")
        self.log.info(f"# SEMANTIC DISTILLATION PIPELINE")
        self.log.info(f"# {datetime.now().isoformat()}")
        self.log.info(f"# Model: {self.model}")
        self.log.info(f"# Feature set: {self.feature_set.name}")
        self.log.info(f"# Dataset: {len(df)} samples")
        self.log.info(f"{'#'*60}\n")

        # Step 1: Extract features
        features_df = self.extract_batch(df, text_col=text_col)

        # Step 2: Build feature matrix
        X, feature_names = self.build_feature_matrix(features_df, log=self.log)

        # Align labels with successfully extracted features
        y = df.set_index(df.get("id", pd.RangeIndex(len(df)))).loc[X.index, label_col]

        # Step 3: Fit and evaluate
        results = self.fit_and_evaluate(X, y, feature_names, log=self.log)

        # Step 4: Residual analysis (generate hypotheses for next iteration)
        residual_analysis = None
        if do_residual_analysis:
            texts = df.set_index(df.get("id", pd.RangeIndex(len(df)))).loc[X.index, text_col]
            residual_analysis = self.analyze_residuals(X, y, texts, feature_names)

        # Calculate costs
        cost_info = self.calculate_cost()

        # Save artifacts
        output = {
            "run_id": self.run_id,
            "model": self.model,
            "feature_set": self.feature_set.name,
            "feature_config": self.feature_config,
            "n_samples": len(df),
            "n_extracted": len(features_df),
            "features": feature_names,
            "results": results,
            "extraction_log": self.extraction_log,
            "cost": cost_info,
        }

        if residual_analysis:
            output["residual_analysis"] = {
                "pattern_summary": residual_analysis.pattern_summary,
                "proposed_features": [h.model_dump() for h in residual_analysis.hypotheses],
            }

        # Save full results
        results_path = save_results(output, self.log_dir, self.run_id)
        self.log.info(f"\nResults saved to {results_path}")

        # Save feature matrix for inspection
        features_path = save_features_csv(features_df, self.log_dir, self.run_id)
        self.log.info(f"Feature matrix saved to {features_path}")

        # Summary
        self.log.info(f"\n{'='*60}")
        self.log.info("SUMMARY")
        self.log.info(f"{'='*60}")
        self.log.info(f"Extracted {len(feature_names)} features from {len(features_df)} posts")
        self.log.info(f"CV Accuracy: {results['cv_accuracy_mean']:.3f} (±{results['cv_accuracy_std']:.3f})")
        self.log.info(f"Non-zero features (L1): {results['n_nonzero_features']}")

        top_features = sorted(results["coefficients"], key=lambda x: abs(x["coefficient"]), reverse=True)[:5]
        self.log.info(f"\nTop 5 features by coefficient magnitude:")
        for f in top_features:
            self.log.info(f"  {f['feature']:40s} coef={f['coefficient']:+.4f}")

        if residual_analysis:
            self.log.info(f"\nNext iteration suggestions:")
            for h in residual_analysis.hypotheses[:3]:
                self.log.info(f"  → Add '{h.feature_name}' ({h.feature_type})")

        # Token usage and cost summary
        self.log.info(f"\n{'='*60}")
        self.log.info("TOKEN USAGE & COST")
        self.log.info(f"{'='*60}")
        self.log.info(f"Model: {self.model}")
        self.log.info(f"Input tokens:  {cost_info['input_tokens']:,}")
        self.log.info(f"Output tokens: {cost_info['output_tokens']:,}")
        self.log.info(f"Total tokens:  {cost_info['total_tokens']:,}")
        self.log.info(f"")
        self.log.info(f"Input cost:  ${cost_info['input_cost_usd']:.4f}")
        self.log.info(f"Output cost: ${cost_info['output_cost_usd']:.4f}")
        self.log.info(f"Total cost:  ${cost_info['total_cost_usd']:.4f}")
        
        # Extrapolate to full dataset
        if len(features_df) > 0:
            cost_per_sample = cost_info['total_cost_usd'] / len(features_df)
            self.log.info(f"")
            self.log.info(f"Cost per sample: ${cost_per_sample:.5f}")
            self.log.info(f"Estimated cost for 5,000 samples: ${cost_per_sample * 5000:.2f}")

        return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Semantic Distillation: LLM feature extraction → interpretable models")
    parser.add_argument("--data", type=str, default="dataset/aita_dataset.csv", help="Path to CSV file with text data")
    parser.add_argument("--sample-size", type=int, default=40, help="Number of samples to use (default: 40, 0 for all)")
    parser.add_argument("--text-col", type=str, default="body", help="Column name for text")
    parser.add_argument("--label-col", type=str, default="verdict", help="Column name for labels")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="OpenAI model to use")
    parser.add_argument("--features", type=str, default=None, help="Path to JSON file with feature config")
    parser.add_argument("--no-residuals", action="store_true", help="Skip residual analysis")
    parser.add_argument("--use-cache", action="store_true", help="Use cached feature extractions (off by default)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache before running")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent extraction requests (default: 20)")
    parser.add_argument("--rpm", type=float, default=500.0, help="Rate limit: requests per minute (default: 500)")
    args = parser.parse_args()

    # Handle cache clearing
    if args.clear_cache:
        cache = FeatureCache()
        count = cache.clear()
        print(f"Cleared {count} cached items")

    # Load feature configuration
    feature_config = None
    if args.features:
        feature_config = FeatureSet.from_json(args.features).to_dict()
        print(f"Loaded feature config from {args.features}")

    # Load data
    df = load_data(args.data, args.text_col, args.label_col)
    if args.data:
        print(f"Loaded data from {args.data}")
    else:
        print("No data file provided, using built-in AITA sample posts")
    
    # Sample if requested
    if args.sample_size > 0 and len(df) > args.sample_size:
        print(f"Sampling {args.sample_size} from {len(df)} rows (stratified by {args.label_col})...")
        # Stratified sample to maintain label distribution
        sampled_indices = []
        total = len(df)
        for label in df[args.label_col].unique():
            label_df = df[df[args.label_col] == label]
            n_to_sample = max(1, int(args.sample_size * len(label_df) / total))
            sampled = label_df.sample(n=min(n_to_sample, len(label_df)), random_state=42)
            sampled_indices.extend(sampled.index.tolist())
        df = df.loc[sampled_indices].reset_index(drop=True)
        # Trim to exact size if needed
        if len(df) > args.sample_size:
            df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df[args.label_col].value_counts().to_string()}")

    # Run pipeline
    distiller = SemanticDistiller(
        feature_config=feature_config,
        model=args.model,
        use_cache=args.use_cache,
        max_concurrent=args.max_concurrent,
        requests_per_minute=args.rpm,
    )
    results = distiller.run(
        df,
        text_col=args.text_col,
        label_col=args.label_col,
        do_residual_analysis=not args.no_residuals,
    )


if __name__ == "__main__":
    main()
