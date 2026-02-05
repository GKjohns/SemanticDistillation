# Semantic Distillation

LLM-powered interpretable feature engineering for text classification.

**The idea:** Use an LLM to extract structured, human-interpretable features from unstructured text, then fit classical statistical models (logistic regression) on those features. You get the semantic understanding of LLMs with the interpretability of traditional ML.

## How It Works

1. **Feature Definition** - Define a set of features you want to extract (e.g., `self_awareness`, `emotional_intensity`, `empathy_shown`)
2. **LLM Extraction** - Use OpenAI's structured outputs API to extract feature values from each text sample
3. **Model Fitting** - Fit a regularized logistic regression (elastic net) on the extracted features
4. **Interpretation** - Examine coefficients to understand what drives predictions
5. **Iteration** - Analyze misclassifications and propose new features to improve accuracy

## Demo: AITA Verdict Prediction

This repo demonstrates the approach on Reddit's "Am I the Asshole?" (AITA) dataset:
- **X** = Post text describing an interpersonal conflict
- **Y** = Crowd verdict (binary: NTA = Not The Asshole, YTA = You're The Asshole)

The raw dataset has 4 labels (NTA, YTA, ESH, NAH), but the data loader converts to binary by default:
- **NAH** ("No Assholes Here") → **NTA** (poster is not at fault)
- **ESH** ("Everyone Sucks Here") → dropped (ambiguous for binary classification)

The model learns which aspects of how people describe conflicts correlate with crowd judgments.

### Important: Crowd Consensus vs Objective Truth

The verdict labels are **not objective moral truth** — they represent the **weighted consensus** of Reddit commenters' votes. The goal is to predict what the crowd will say, not what is objectively "right."

This is a subtle but important distinction. The crowd may:
- Respond to certain writing styles or narrative framings
- Have systematic biases (e.g., sympathizing with certain demographics)
- Be influenced by how sympathetically the poster presents themselves
- Follow patterns that don't align with an objective moral assessment

The features we extract should capture what **actually influences crowd perception**, which may differ from what "should" matter in an objective sense. This makes the problem more tractable and the learned patterns more interpretable — we're modeling human judgment, not implementing a moral philosophy.

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/SemanticDistillation.git
cd SemanticDistillation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=sk-...
```

## Quick Start

```bash
# Run on a small sample (40 posts by default)
python -m src.semantic_distillation

# Run with more samples
python -m src.semantic_distillation --sample-size 100

# Use cached extractions to avoid re-calling the API
python -m src.semantic_distillation --use-cache

# Parallel extraction with rate limiting (default: 20 concurrent, 500 rpm)
python -m src.semantic_distillation --max-concurrent 20 --rpm 500

# Use a different model
python -m src.semantic_distillation --model gpt-5

# Skip residual analysis
python -m src.semantic_distillation --no-residuals
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | `dataset/aita_dataset.csv` | Path to CSV file with text data |
| `--sample-size` | 40 | Number of samples to use (0 for all) |
| `--text-col` | `body` | Column name for text |
| `--label-col` | `verdict` | Column name for labels |
| `--model` | `gpt-5-mini` | OpenAI model to use |
| `--features` | None | Path to custom feature config JSON |
| `--use-cache` | False | Use cached feature extractions |
| `--clear-cache` | False | Clear cache before running |
| `--max-concurrent` | 20 | Max concurrent extraction requests |
| `--rpm` | 500 | Rate limit (requests per minute) |
| `--no-residuals` | False | Skip residual analysis |

## Iterative Feature Improvement

The `iterative_improvement.py` script automates the feature engineering loop:

```bash
# Run 5 iterations of feature improvement
python iterative_improvement.py --iterations 5

# Customize the process
python iterative_improvement.py \
    --iterations 3 \
    --sample-size 50 \
    --max-swaps 3 \
    --analysis-model gpt-5 \
    --extraction-model gpt-5-mini
```

Each iteration:
1. Extracts features from text samples and fits the model
2. Analyzes correctly vs incorrectly classified samples
3. Uses an LLM to evaluate each feature's usefulness
4. Proposes new features to replace underperformers
5. Saves results to `iteration_logs/`

### Validation Split Mode (Recommended)

By default, the same data sample is used for all iterations, which can lead to overfitting to that specific sample. For more robust feature selection, use the `--validation-split` flag:

```bash
# Enable train/validation/test split
python iterative_improvement.py --iterations 10 --validation-split

# Customize split sizes
python iterative_improvement.py --iterations 10 --validation-split \
    --sample-size 100 \
    --val-size 150 \
    --test-size 300 \
    --n-folds 10
```

With `--validation-split`:
- **Test set** (default 200): Held out until final evaluation, never seen during feature development
- **Validation set** (default 100): Fixed across iterations, used to evaluate proposed features
- **Train pool**: Remaining data, split into rotating folds for each iteration

This ensures features generalize well rather than overfitting to a single sample. The best model (by validation accuracy) is automatically evaluated on the held-out test set at the end.

### Statistical Interpretation

When interpreting results from iterative feature improvement, keep in mind:

- **Validation accuracy** is used repeatedly to evaluate and select features. This means it's **optimistically biased** — features that survive are ones that happen to work well on that specific validation sample. Treat validation accuracy as showing the *optimization trajectory* (relative improvement), not absolute generalization.

- **Test accuracy** is the unbiased estimate of real-world performance. The test set is held out and never seen during feature selection, so it gives an honest measure of how well the final features generalize.

- **Sample sizes** affect confidence intervals. With 200-300 test samples, expect roughly ±5-6 percentage points of uncertainty (95% CI) on accuracy estimates.

In short: use validation accuracy to track progress during development, but report test accuracy as your final result.

### Iterative Improvement Options

| Option | Default | Description |
|--------|---------|-------------|
| `--iterations`, `-n` | 5 | Number of improvement iterations |
| `--sample-size`, `-s` | 50 | Train samples per iteration |
| `--max-swaps` | 3 | Max features to swap per iteration |
| `--data` | `dataset/aita_dataset.csv` | Path to data file |
| `--analysis-model` | `gpt-5` | Model for strategic analysis |
| `--extraction-model` | `gpt-5-mini` | Model for feature extraction |
| `--output-dir` | `iteration_logs` | Directory for output files |
| `--validation-split`, `-v` | False | Enable train/val/test split |
| `--val-size` | 100 | Size of fixed validation set |
| `--test-size` | 200 | Size of held-out test set |
| `--n-folds` | 5 | Number of train folds to rotate |

## Project Structure

```
SemanticDistillation/
├── src/                           # Core library
│   ├── semantic_distillation.py   # Main pipeline (extraction + modeling)
│   ├── features.py                # Feature set definitions
│   ├── schemas.py                 # Pydantic models for structured extraction
│   ├── data.py                    # Data loading utilities + sample data
│   └── utils.py                   # Logging, caching, rate limiting
├── dataset/                       # AITA dataset
│   ├── aita_dataset.csv           # Minimal dataset (id, body, verdict)
│   ├── aita_dataset_full.csv      # Full dataset with metadata
│   └── README.md                  # Dataset documentation
├── build_dataset.py               # Script to build dataset from SQLite source
├── iterative_improvement.py       # Automated feature improvement loop
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Default Feature Set

The default feature set (`DEFAULT_FEATURES`) includes 8 interpretable features:

| Feature | Type | Description |
|---------|------|-------------|
| `self_awareness` | scale (1-5) | Acknowledgment of own role/potential wrongdoing |
| `empathy_shown` | scale (1-5) | Understanding of the other party's perspective |
| `emotional_intensity` | scale (1-5) | Calm/neutral vs highly emotional tone |
| `harm_caused` | scale (1-5) | Severity of harm (minor inconvenience to serious) |
| `provocation_received` | scale (1-5) | Whether poster was provoked first |
| `proportionality` | scale (1-5) | Response appropriateness to situation |
| `relationship_closeness` | scale (1-5) | Strangers to very close relationship |
| `power_dynamic` | scale (1-5) | Who has more power in the relationship |

Additional feature sets are available in `src/features.py`:
- `FULL_FEATURES` - 15 features including bools and categoricals
- `MINIMAL_FEATURES` - 3 features for quick testing

## Custom Feature Sets

Define your own features in JSON:

```json
{
  "name": "MyFeatures",
  "description": "Custom features for my use case",
  "features": [
    {
      "name": "sentiment_score",
      "type": "scale",
      "description": "1=very negative, 5=very positive",
      "min": 1,
      "max": 5
    },
    {
      "name": "contains_apology",
      "type": "bool",
      "description": "Does the text contain an apology?"
    },
    {
      "name": "primary_emotion",
      "type": "categorical",
      "description": "The dominant emotion expressed",
      "values": ["anger", "sadness", "fear", "joy", "neutral"]
    }
  ]
}
```

Then run with:

```bash
python -m src.semantic_distillation --features my_features.json
```

## Example Output

```
COEFFICIENT TABLE (sorted by importance)
============================================================
  + proportionality                          coef=+0.8234  OR=2.278
  + empathy_shown                            coef=+0.6891  OR=1.992
  - emotional_intensity                      coef=-0.4523  OR=0.636
  + self_awareness                           coef=+0.3912  OR=1.479
  ...

Cross-validation accuracy: 0.725 (±0.045)
```

**Interpretation:** Posts where the author shows proportional responses and empathy are more likely to receive NTA verdicts, while high emotional intensity correlates with YTA verdicts.

## API Cost Estimation

The pipeline tracks token usage and estimates costs. Supported models and pricing (per 1M tokens):

| Model | Input | Output |
|-------|-------|--------|
| `gpt-5-mini` | $0.25 | $2.00 |
| `gpt-5` | $1.25 | $10.00 |
| `gpt-5.2` | $1.75 | $14.00 |
| `gpt-5.2-pro` | $21.00 | $168.00 |

Typical cost for 100 samples with `gpt-5-mini`: ~$0.05-0.10

## Caching

Feature extractions can be cached to avoid redundant API calls:

```bash
# Enable caching
python -m src.semantic_distillation --use-cache

# Clear cache
python -m src.semantic_distillation --clear-cache
```

Cache files are stored in `cache/` as JSON files keyed by MD5 hash of the input text.

## Related Work

This implementation draws inspiration from recent work on LLM-powered feature engineering:

**FELIX: Automatic and Interpretable Feature Engineering Using LLMs**
Simon Malberg, Edoardo Mosca, Georg Groh. ECML PKDD 2024.
[Paper](https://link.springer.com/chapter/10.1007/978-3-031-70359-1_14) | [Code](https://github.com/simonmalberg/felix)

FELIX uses LLMs to automatically generate human-interpretable features from text, demonstrating that structured feature extraction can outperform both traditional methods (TF-IDF) and LLM embeddings on text classification tasks.

**CAAFE: Context-Aware Automated Feature Engineering**
Noah Hollmann, Samuel Müller, Frank Hutter. NeurIPS 2023.
[arXiv:2305.03403](https://arxiv.org/abs/2305.03403) | [Code](https://github.com/automl/CAAFE)

CAAFE uses LLMs to iteratively generate semantically meaningful features for tabular datasets based on dataset descriptions, producing both Python code and explanations for generated features.

## License

MIT
