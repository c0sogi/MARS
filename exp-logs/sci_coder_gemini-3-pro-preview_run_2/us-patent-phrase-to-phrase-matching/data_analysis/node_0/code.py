import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.feature_extraction.text import CountVectorizer
import random

# --- Configuration ---
DATA_PATH = "./metadata/train.csv"
SEED = 42


# --- Seeding ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)


def analyze_text_column(df, col_name):
    """Calculates length statistics for a text column."""
    char_lens = df[col_name].astype(str).apply(len)
    word_lens = df[col_name].astype(str).apply(lambda x: len(x.split()))
    return char_lens, word_lens


def get_jaccard_sim(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


def main():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)

    # ==========================================
    # SECTION 1: TARGET VARIABLE ANALYSIS
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")
    target = df["score"]

    # Distribution stats
    print(f"Mean: {target.mean():.4f}")
    print(f"Std Dev: {target.std():.4f}")
    print(f"Min: {target.min():.4f}")
    print(f"Max: {target.max():.4f}")

    # Normality check
    target_skew = skew(target)
    target_kurt = kurtosis(target)
    print(f"Skewness: {target_skew:.4f}")
    print(f"Kurtosis: {target_kurt:.4f}")

    # Discrete value distribution
    print("\nScore Value Distribution:")
    value_counts = target.value_counts(normalize=True).sort_index()
    for score_val, prop in value_counts.items():
        print(f"Score {score_val:.2f}: {prop:.4f}")
    print("-" * 30)

    # ==========================================
    # SECTION 2: INPUT DATA ANALYSIS (TEXT)
    # ==========================================
    print("INPUT DATA ANALYSIS (TEXT)")

    text_cols = ["anchor", "target"]

    # Length Analysis
    for col in text_cols:
        char_lens, word_lens = analyze_text_column(df, col)
        print(f"\nColumn: '{col}'")
        print(f"  Mean Char Length: {char_lens.mean():.4f}")
        print(f"  Max Char Length:  {char_lens.max():.4f}")
        print(f"  Mean Word Length: {word_lens.mean():.4f}")
        print(f"  Max Word Length:  {word_lens.max():.4f}")

    # Vocabulary Analysis
    # Combine all text to check global vocabulary
    all_text = pd.concat([df["anchor"], df["target"]]).astype(str)
    vectorizer = CountVectorizer()
    vectorizer.fit(all_text)
    vocab_size = len(vectorizer.vocabulary_)

    print(f"\nGlobal Vocabulary Size: {vocab_size}")
    print("-" * 30)

    # ==========================================
    # SECTION 3: INPUT DATA ANALYSIS (CATEGORICAL)
    # ==========================================
    print("INPUT DATA ANALYSIS (CATEGORICAL)")

    cat_col = "context"
    n_unique = df[cat_col].nunique()
    print(f"Column: '{cat_col}'")
    print(f"  Cardinality: {n_unique}")

    # Check for rare labels (< 1%)
    counts = df[cat_col].value_counts(normalize=True)
    rare_labels = counts[counts < 0.01]
    print(f"  Rare Labels (< 1% freq): {len(rare_labels)} out of {n_unique}")
    print("-" * 30)

    # ==========================================
    # SECTION 4: FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Generate Meta-Features
    df["anchor_len"] = df["anchor"].astype(str).apply(len)
    df["target_len"] = df["target"].astype(str).apply(len)
    df["len_diff"] = (df["anchor_len"] - df["target_len"]).abs()
    df["word_count_diff"] = (
        df["anchor"].astype(str).apply(lambda x: len(x.split()))
        - df["target"].astype(str).apply(lambda x: len(x.split()))
    ).abs()

    # Simple Jaccard Similarity as a proxy for semantic overlap
    df["jaccard_sim"] = df.apply(
        lambda row: get_jaccard_sim(str(row["anchor"]), str(row["target"])), axis=1
    )

    meta_features = [
        "anchor_len",
        "target_len",
        "len_diff",
        "word_count_diff",
        "jaccard_sim",
    ]

    print("\nCorrelation with Target (Score):")
    correlations = df[meta_features].corrwith(df["score"])
    for feat, corr in correlations.items():
        print(f"  {feat}: {corr:.4f}")

    # Categorical Relationship (Context vs Score)
    # Check if some contexts have significantly different mean scores
    print("\nContext vs Score Analysis:")
    context_stats = df.groupby("context")["score"].agg(["mean", "count"])
    # Sort by count to see major categories
    top_contexts = context_stats.sort_values("count", ascending=False).head(5)
    print("  Top 5 Contexts by Frequency (Mean Score):")
    for ctx, row in top_contexts.iterrows():
        print(f"    {ctx} (n={int(row['count'])}): {row['mean']:.4f}")

    # Check for contexts with highest/lowest mean scores (min 50 samples to be significant)
    significant_contexts = context_stats[context_stats["count"] > 50]
    if not significant_contexts.empty:
        highest_ctx = significant_contexts["mean"].idxmax()
        lowest_ctx = significant_contexts["mean"].idxmin()
        print(
            f"  Context with Highest Mean Score (>50 samples): {highest_ctx} ({significant_contexts.loc[highest_ctx, 'mean']:.4f})"
        )
        print(
            f"  Context with Lowest Mean Score (>50 samples):  {lowest_ctx} ({significant_contexts.loc[lowest_ctx, 'mean']:.4f})"
        )


if __name__ == "__main__":
    main()
