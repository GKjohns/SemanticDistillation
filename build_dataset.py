"""
Build a tractable AITA dataset from the SQLite database.

This script:
1. Samples posts at the submission level first
2. Filters real AITA posts (excludes meta posts, empty bodies, etc.)
3. Gets comments for sampled posts only
4. Extracts verdict labels from comments (NTA, YTA, ESH, NAH)
5. Weights verdicts by comment score
6. Determines winning verdict per submission
7. Outputs CSV ready for semantic_distillation.py

Usage:
    python build_dataset.py [--sample SIZE] [--output PATH]
"""

import re
import sqlite3
import argparse
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Verdict patterns - looking for labels at start of comment or standalone
# AITA verdicts: NTA (Not The Asshole), YTA (You're The Asshole), 
#                ESH (Everyone Sucks Here), NAH (No Assholes Here)
VERDICT_PATTERN = re.compile(
    r'\b(NTA|YTA|ESH|NAH)\b',
    re.IGNORECASE
)

# Meta post patterns to exclude
META_PATTERNS = [
    r'Monthly Open Forum',
    r'META',
    r'\[META\]',
    r'Rule \d+',
    r'Welcome to',
    r'Modpost',
]
META_REGEX = re.compile('|'.join(META_PATTERNS), re.IGNORECASE)


def extract_verdict(message: str) -> str | None:
    """Extract the first verdict label from a comment message."""
    if not message:
        return None
    
    # Look for verdict in first 200 chars (usually at the start)
    search_text = message[:200]
    match = VERDICT_PATTERN.search(search_text)
    
    if match:
        return match.group(1).upper()
    return None


def is_valid_submission(title: str, selftext: str) -> bool:
    """Check if a submission is a real AITA post (not meta, has content)."""
    if not title or not selftext:
        return False
    
    # Skip meta posts
    if META_REGEX.search(title):
        return False
    
    # Skip deleted/removed posts
    if selftext in ['[deleted]', '[removed]', '']:
        return False
    
    # Require minimum length (actual stories should be substantial)
    if len(selftext) < 100:
        return False
    
    return True


def build_dataset(
    db_path: str,
    sample_size: int = 5000,
    top_k_comments: int = 50,
    min_verdict_comments: int = 3,
    min_score_total: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the AITA dataset from SQLite database.
    
    Approach: Sample posts first, then get comments for those posts only.
    
    Args:
        db_path: Path to SQLite database
        sample_size: Number of posts to sample
        top_k_comments: Get top K comments by score for each post (0 for all)
        min_verdict_comments: Minimum number of comments with verdicts required
        min_score_total: Minimum total score across verdict comments
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: id, body, verdict
    """
    random.seed(seed)
    conn = sqlite3.connect(db_path)
    
    # Step 1: Load all valid submissions and sample at the post level
    print("Loading submissions...")
    all_submissions = []
    query = "SELECT id, submission_id, title, selftext, score FROM submission"
    
    for row in conn.execute(query):
        sub_id, reddit_id, title, selftext, score = row
        if is_valid_submission(title, selftext):
            all_submissions.append({
                'id': sub_id,
                'reddit_id': reddit_id,  # This is the string ID used in comments
                'title': title,
                'body': selftext,
                'score': score,
            })
    
    print(f"Found {len(all_submissions)} valid submissions")
    
    # Sample more than we need since some won't have enough verdict comments
    oversample_factor = 1.5
    sample_count = min(len(all_submissions), int(sample_size * oversample_factor))
    sampled_submissions = random.sample(all_submissions, sample_count)
    
    print(f"Sampled {len(sampled_submissions)} posts (oversampling to ensure {sample_size} final)")
    
    # Build lookup by reddit_id (the string ID that comments reference)
    submissions_by_reddit_id = {s['reddit_id']: s for s in sampled_submissions}
    reddit_ids = set(submissions_by_reddit_id.keys())
    
    # Step 2: Get comments for sampled posts only
    print(f"Loading comments for sampled posts (top {top_k_comments} per post)...")
    
    # Use a query that filters to only our sampled posts
    # Build placeholders for the IN clause
    placeholders = ','.join(['?' for _ in reddit_ids])
    
    if top_k_comments > 0:
        # Get top K comments per post using a window function
        query = f"""
            SELECT submission_id, message, score FROM (
                SELECT submission_id, message, score,
                       ROW_NUMBER() OVER (PARTITION BY submission_id ORDER BY score DESC) as rn
                FROM comment
                WHERE submission_id IN ({placeholders})
            ) WHERE rn <= ?
        """
        params = list(reddit_ids) + [top_k_comments]
    else:
        # Get all comments
        query = f"""
            SELECT submission_id, message, score
            FROM comment
            WHERE submission_id IN ({placeholders})
        """
        params = list(reddit_ids)
    
    # Process comments and aggregate verdicts
    verdict_scores = defaultdict(lambda: defaultdict(int))
    verdict_counts = defaultdict(lambda: defaultdict(int))
    comment_count = 0
    
    for row in conn.execute(query, params):
        reddit_id, message, score = row
        comment_count += 1
        
        verdict = extract_verdict(message)
        if verdict:
            # Use max(score, 0) to avoid negative scores dragging things down
            verdict_scores[reddit_id][verdict] += max(score, 0)
            verdict_counts[reddit_id][verdict] += 1
    
    print(f"Processed {comment_count:,} comments")
    conn.close()
    
    # Step 3: Determine winning verdict for each submission
    print("Determining verdicts...")
    results = []
    
    for reddit_id, sub_data in submissions_by_reddit_id.items():
        scores = verdict_scores.get(reddit_id, {})
        counts = verdict_counts.get(reddit_id, {})
        
        if not scores:
            continue
        
        # Require minimum engagement
        total_comments = sum(counts.values())
        total_score = sum(scores.values())
        
        if total_comments < min_verdict_comments:
            continue
        if total_score < min_score_total:
            continue
        
        # Find winning verdict (highest total score)
        winning_verdict = max(scores.keys(), key=lambda v: scores[v])
        winning_score = scores[winning_verdict]
        
        # Calculate confidence (winning score / total score)
        confidence = winning_score / total_score if total_score > 0 else 0
        
        results.append({
            'id': reddit_id,
            'body': sub_data['body'],
            'verdict': winning_verdict,
            'submission_score': sub_data['score'],
            'verdict_score': winning_score,
            'total_verdict_score': total_score,
            'confidence': confidence,
            'num_verdict_comments': total_comments,
        })
    
    print(f"Found {len(results)} submissions with clear verdicts")
    
    df = pd.DataFrame(results)
    
    # Trim to exact sample size if we have more than needed
    if len(df) > sample_size:
        print(f"Trimming to {sample_size} posts (stratified by verdict)...")
        
        # Stratified sampling - sample proportionally from each verdict
        sampled_indices = []
        total = len(df)
        for verdict in df['verdict'].unique():
            verdict_df = df[df['verdict'] == verdict]
            n_to_sample = max(1, int(sample_size * len(verdict_df) / total))
            sampled = verdict_df.sample(n=min(n_to_sample, len(verdict_df)), random_state=seed)
            sampled_indices.extend(sampled.index.tolist())
        
        df = df.loc[sampled_indices]
        
        # Ensure we hit exactly sample_size
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=seed)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Build AITA dataset from SQLite")
    parser.add_argument(
        "--db", 
        type=str, 
        default="dataset/AmItheAsshole.sqlite",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--sample", 
        type=int, 
        default=5000,
        help="Number of posts to sample (default: 5000)"
    )
    parser.add_argument(
        "--top-k-comments",
        type=int,
        default=50,
        help="Get top K comments per post (default: 50, 0 for all)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="dataset/aita_dataset.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=3,
        help="Minimum verdict comments per post (default: 3)"
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=5,
        help="Minimum total verdict score (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 1
    
    df = build_dataset(
        str(db_path),
        sample_size=args.sample,
        top_k_comments=args.top_k_comments,
        min_verdict_comments=args.min_comments,
        min_score_total=args.min_score,
        seed=args.seed,
    )
    
    # Print statistics
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total posts: {len(df)}")
    print(f"\nVerdict distribution:")
    print(df['verdict'].value_counts().to_string())
    print(f"\nConfidence stats:")
    print(df['confidence'].describe().to_string())
    print(f"\nBody length stats:")
    df['body_length'] = df['body'].str.len()
    print(df['body_length'].describe().to_string())
    
    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save minimal version (just id, body, verdict) for the semantic distillation script
    minimal_df = df[['id', 'body', 'verdict']]
    minimal_df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1_000_000:.2f} MB")
    
    # Also save full version with metadata
    full_output = output_path.with_stem(output_path.stem + '_full')
    df.to_csv(full_output, index=False)
    print(f"Full dataset (with metadata) saved to {full_output}")
    print(f"File size: {full_output.stat().st_size / 1_000_000:.2f} MB")
    
    return 0


if __name__ == "__main__":
    exit(main())
