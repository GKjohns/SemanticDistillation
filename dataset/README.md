# AITA Dataset

A processed dataset of Reddit "Am I the Asshole?" (AITA) posts with crowd-sourced verdicts, designed for use with the semantic distillation pipeline.

## Source Data

**Original dataset:** [Reddit AITA Dataset on Kaggle](https://www.kaggle.com/datasets/jianloongliew/reddit)

The raw SQLite database (`AmItheAsshole.sqlite`, ~1GB) contains:
- ~31,000 submissions (post text)
- ~9 million comments (including verdict labels)

## Processed Dataset

| File | Description | Size |
|------|-------------|------|
| `aita_dataset.csv` | Minimal dataset (id, body, verdict) | ~10 MB |
| `aita_dataset_full.csv` | Full dataset with metadata | ~10 MB |

### Columns

**Minimal (`aita_dataset.csv`):**
- `id` - Reddit submission ID (string)
- `body` - Post text (the AITA story)
- `verdict` - Crowd verdict (NTA, YTA after binary conversion; raw file has ESH, NAH too)

**Full (`aita_dataset_full.csv`):**
- All columns above, plus:
- `submission_score` - Reddit score of the post
- `verdict_score` - Total score of comments with winning verdict
- `total_verdict_score` - Total score across all verdict comments
- `confidence` - Ratio of winning verdict score to total (0-1)
- `num_verdict_comments` - Number of comments containing verdicts

### Verdict Labels (Raw)

The raw dataset contains four verdict labels:

| Label | Meaning |
|-------|---------|
| NTA | "Not The Asshole" - Crowd says OP is in the right |
| YTA | "You're The Asshole" - Crowd says OP is in the wrong |
| ESH | "Everyone Sucks Here" - Crowd says both parties are wrong |
| NAH | "No Assholes Here" - Crowd says no one is wrong |

### Binary Classification (Default)

By default, the data loader converts to binary classification for cleaner modeling:

| Original | Mapped To | Rationale |
|----------|-----------|-----------|
| NTA | NTA | Poster is not at fault |
| NAH | NTA | "No one is wrong" → poster is not at fault |
| YTA | YTA | Poster is at fault |
| ESH | *dropped* | "Everyone is wrong" is ambiguous for binary |

This reduces the task to a clean binary question: **"Is the poster at fault?"**

To disable this and keep all 4 labels, use `binary_labels=False` when calling `load_data()`.

**Important:** These verdicts represent **crowd consensus**, not objective moral truth. The label is determined by the weighted aggregate of Reddit commenters' votes. When building models on this data, the goal is to predict what the crowd will say, not what is objectively "right." The crowd may have biases, respond to narrative framing, or follow patterns that differ from an objective assessment.

## How Verdicts Were Determined

The `build_dataset.py` script processes the raw SQLite database:

1. **Post-level sampling**: Randomly samples valid submissions (excluding meta posts, deleted content, and posts < 100 chars)

2. **Comment retrieval**: For each sampled post, retrieves the top K comments by score (default: 50)

3. **Verdict extraction**: Parses verdict labels (NTA/YTA/ESH/NAH) from the first 200 characters of each comment using regex (`\b(NTA|YTA|ESH|NAH)\b`, case-insensitive)

4. **Score weighting**: Aggregates scores by verdict type per post (negative scores treated as 0)

5. **Winner selection**: The verdict with the highest total score wins

6. **Quality filtering**: Requires at least N verdict comments and M total score per post (configurable)

7. **Final sampling**: Stratified sample to target size, maintaining verdict distribution

## Rebuilding the Dataset

To rebuild or customize the dataset:

```bash
# Ensure you have the source database
# Download from: https://www.kaggle.com/datasets/jianloongliew/reddit
# Place at: dataset/AmItheAsshole.sqlite

# Create virtual environment (if needed)
python3 -m venv venv
source venv/bin/activate
pip install pandas

# Build dataset with defaults (5000 posts, top 50 comments each)
python build_dataset.py

# Customize parameters
python build_dataset.py \
    --sample 10000 \
    --top-k-comments 100 \
    --min-comments 5 \
    --min-score 10 \
    --output dataset/aita_custom.csv
```

### Build Script Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | `dataset/AmItheAsshole.sqlite` | Path to SQLite database |
| `--sample` | 5000 | Number of posts to sample |
| `--top-k-comments` | 50 | Top K comments per post (0 for all) |
| `--min-comments` | 3 | Minimum verdict comments required |
| `--min-score` | 5 | Minimum total verdict score required |
| `--seed` | 42 | Random seed for reproducibility |
| `--output` | `dataset/aita_dataset.csv` | Output path for minimal CSV |

The script automatically generates two files:
- Minimal dataset at the specified output path
- Full dataset with `_full` suffix (e.g., `aita_dataset_full.csv`)

## Usage with Semantic Distillation

```bash
# Basic usage (uses dataset by default)
python -m src.semantic_distillation

# Explicit path
python -m src.semantic_distillation \
    --data dataset/aita_dataset.csv \
    --text-col body \
    --label-col verdict

# With iterative improvement
python iterative_improvement.py \
    --data dataset/aita_dataset.csv \
    --iterations 5 \
    --validation-split
```

## Dataset Statistics (default build)

**Raw dataset:**
- **Total posts:** ~5,000
- **Verdict distribution:** NTA (~78%), YTA (~17%), ESH (~2%), NAH (~2%)

**After binary conversion (default):**
- **Total posts:** ~4,884 (ESH samples dropped)
- **Verdict distribution:** NTA (~83%), YTA (~17%)
- **Average body length:** ~2,000 characters
- **Average confidence:** ~0.95 (high agreement on winning verdict)

Note: The class imbalance (NTA dominant) is handled by the modeling pipeline using `class_weight="balanced"` in logistic regression.
