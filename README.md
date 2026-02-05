# Semantic Distillation

LLM-powered interpretable feature engineering for text classification.

**The idea:** Use an LLM to extract structured, human-interpretable features from unstructured text, then fit classical statistical models (logistic regression) on those features. You get the semantic understanding of LLMs with the interpretability of traditional ML.

## How It Works

1. **Feature Definition** - Define a set of features you want to extract (e.g., "self_awareness", "emotional_intensity", "empathy_shown")
2. **LLM Extraction** - Use structured outputs to extract feature values from each text sample
3. **Model Fitting** - Fit a regularized logistic regression on the extracted features
4. **Interpretation** - Examine coefficients to understand what drives predictions
5. **Iteration** - Analyze misclassifications and propose new features to improve accuracy

## Demo: AITA Verdict Prediction

This repo demonstrates the approach on Reddit's "Am I the Asshole?" (AITA) dataset:
- **X** = Post text describing an interpersonal conflict
- **Y** = Crowd verdict (NTA = Not The Asshole, YTA = You're The Asshole)

The model learns which aspects of how people describe conflicts correlate with crowd judgments.

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

# Parallel extraction with rate limiting
python -m src.semantic_distillation --max-concurrent 20 --rpm 500
```

## Iterative Feature Improvement

The `iterative_improvement.py` script automates the feature engineering loop:

```bash
# Run 5 iterations of feature improvement
python iterative_improvement.py --iterations 5

# Customize the process
python iterative_improvement.py \
    --iterations 3 \
    --sample-size 50 \
    --max-swaps 4 \
    --analysis-model gpt-5 \
    --extraction-model gpt-5-mini
```

Each iteration:
1. Extracts features and fits the model
2. Analyzes correctly vs incorrectly classified samples
3. Uses an LLM to evaluate feature usefulness
4. Proposes new features to replace underperformers
5. Saves results to `iteration_logs/`

## Project Structure

```
SemanticDistillation/
├── src/                        # Core library
│   ├── semantic_distillation.py   # Main pipeline
│   ├── features.py                # Feature set definitions
│   ├── schemas.py                 # Pydantic models for structured extraction
│   ├── data.py                    # Data loading utilities
│   └── utils.py                   # Logging, caching, rate limiting
├── dataset/                    # AITA dataset
│   ├── aita_dataset.csv           # Minimal dataset (id, body, verdict)
│   ├── aita_dataset_full.csv      # Full dataset with metadata
│   └── README.md                  # Dataset documentation
├── build_dataset.py            # Script to build dataset from source
├── iterative_improvement.py    # Automated feature improvement loop
└── requirements.txt
```

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
  + proportionality_of_response              coef=+0.8234  OR=2.278
  + empathy_shown                            coef=+0.6891  OR=1.992
  - emotional_intensity                      coef=-0.4523  OR=0.636
  + self_awareness                           coef=+0.3912  OR=1.479
  ...

Cross-validation accuracy: 0.725 (±0.045)
```

Interpretation: Posts where the author shows proportional responses and empathy are more likely to receive NTA verdicts, while high emotional intensity correlates with YTA verdicts.

## Prior Art

This implementation draws inspiration from:
- **FELIX** (Malberg et al. 2024) - LLM-based feature extraction for tabular data
- **CAAFE** (Hollmann et al. 2024) - Context-aware automated feature engineering

## License

MIT
