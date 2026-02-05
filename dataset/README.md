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
- `id` - Reddit submission ID
- `body` - Post text (the AITA story)
- `verdict` - Crowd verdict (NTA, YTA, ESH, NAH)

**Full (`aita_dataset_full.csv`):**
- All columns above, plus:
- `submission_score` - Reddit score of the post
- `verdict_score` - Total score of comments with winning verdict
- `total_verdict_score` - Total score across all verdict comments
- `confidence` - Ratio of winning verdict score to total (0-1)
- `num_verdict_comments` - Number of comments containing verdicts

### Verdict Labels

| Label | Meaning |
|-------|---------|
| NTA | "Not The Asshole" - OP is in the right |
| YTA | "You're The Asshole" - OP is in the wrong |
| ESH | "Everyone Sucks Here" - Both parties are wrong |
| NAH | "No Assholes Here" - No one is wrong |

## How Verdicts Were Determined

1. **Post-level sampling**: Randomly sampled ~7,500 valid submissions (excluding meta posts, deleted content, and posts < 100 chars)

2. **Comment retrieval**: For each sampled post, retrieved the top 50 comments by score

3. **Verdict extraction**: Parsed verdict labels (NTA/YTA/ESH/NAH) from the first 200 characters of each comment using regex

4. **Score weighting**: Aggregated scores by verdict type per post (negative scores treated as 0)

5. **Winner selection**: The verdict with the highest total score wins

6. **Quality filtering**: Required at least 3 verdict comments and 5 total score per post

7. **Final sampling**: Stratified sample down to ~5,000 posts, maintaining verdict distribution

## Rebuilding the Dataset

To rebuild or customize the dataset:

```bash
# Ensure you have the source database
# Download from: https://www.kaggle.com/datasets/jianloongliew/reddit

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
| `--sample` | 5000 | Number of posts to sample |
| `--top-k-comments` | 50 | Top K comments per post (0 for all) |
| `--min-comments` | 3 | Minimum verdict comments required |
| `--min-score` | 5 | Minimum total verdict score required |
| `--seed` | 42 | Random seed for reproducibility |
| `--output` | `dataset/aita_dataset.csv` | Output path |

## Usage with Semantic Distillation

```bash
python -m src.semantic_distillation \
    --data dataset/aita_dataset.csv \
    --text-col body \
    --label-col verdict
```

## Dataset Statistics (default build)

- **Total posts:** ~5,000
- **Verdict distribution:** NTA (78%), YTA (17%), ESH (2%), NAH (2%)
- **Average body length:** ~2,000 characters
- **Average confidence:** 0.95 (high agreement)
