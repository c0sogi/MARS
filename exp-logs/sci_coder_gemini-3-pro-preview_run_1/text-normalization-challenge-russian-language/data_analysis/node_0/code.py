import pandas as pd
import numpy as np
import random
import os
import sys


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def main():
    seed_everything()

    # --- Configuration ---
    DATA_PATH = "./metadata/train.csv"

    # --- Load Data ---
    # We specify dtype=object for text columns to avoid pandas inferring types incorrectly
    # (e.g. interpreting "123" as int instead of string token)
    try:
        df = pd.read_csv(
            DATA_PATH, dtype={"before": object, "after": object, "class": object}
        )
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}")
        return

    # Handle NaNs that might appear if the token text was literal "nan" or "null"
    df["before"] = df["before"].fillna("")
    df["after"] = df["after"].fillna("")
    df["class"] = df["class"].fillna("UNKNOWN")

    # --- 1. Target Variable Analysis ---
    print("TARGET VARIABLE ANALYSIS")

    # The 'target' in this sequence-to-sequence task is the 'after' column,
    # but the 'class' column is the primary structural target for understanding the data distribution.

    # Class Distribution
    class_counts = df["class"].value_counts()
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {len(class_counts)}")
    print("\nClass Distribution (Top 10):")
    for cls, count in class_counts.head(10).items():
        ratio = count / total_samples
        print(f"  {cls:<15} : {count:>8} ({ratio:.4f})")

    # Imbalance Check
    # In text normalization, 'PLAIN' and 'PUNCT' usually dominate.
    top_class = class_counts.index[0]
    top_ratio = class_counts.iloc[0] / total_samples
    print(f"\nDominant Class: {top_class} ({top_ratio:.4f})")

    # Target Change Analysis (How often does normalization actually happen?)
    # This is effectively the "positive" class for a detection model
    df["is_changed"] = (df["before"] != df["after"]).astype(int)
    change_rate = df["is_changed"].mean()
    print(f"Global Normalization Rate (before != after): {change_rate:.4f}")

    # --- 2. Input Data Analysis (Text Modality) ---
    print("\nINPUT DATA ANALYSIS (TEXT)")

    # Length Analysis (Character counts)
    df["len_before"] = df["before"].astype(str).str.len()
    df["len_after"] = df["after"].astype(str).str.len()

    print("Token Length Statistics (Characters - 'before'):")
    print(f"  Mean: {df['len_before'].mean():.4f}")
    print(f"  Std : {df['len_before'].std():.4f}")
    print(f"  Min : {df['len_before'].min():.4f}")
    print(f"  Max : {df['len_before'].max():.4f}")

    print("\nToken Length Statistics (Characters - 'after'):")
    print(f"  Mean: {df['len_after'].mean():.4f}")
    print(f"  Std : {df['len_after'].std():.4f}")
    print(f"  Max : {df['len_after'].max():.4f}")

    # Expansion/Contraction Ratio
    # Avoid division by zero
    valid_len_mask = df["len_before"] > 0
    expansion_ratio = (
        df.loc[valid_len_mask, "len_after"] / df.loc[valid_len_mask, "len_before"]
    ).mean()
    print(f"\nMean Expansion Ratio (len_after / len_before): {expansion_ratio:.4f}")

    # Vocabulary Analysis
    # Since the dataset is large, we estimate unique counts or use fast operations
    unique_tokens_in = df["before"].nunique()
    unique_tokens_out = df["after"].nunique()

    print(f"\nVocabulary Size (Input 'before'): {unique_tokens_in}")
    print(f"Vocabulary Size (Target 'after'): {unique_tokens_out}")

    # Character Set Analysis (Basic check for non-cyrillic)
    # We take a sample to speed up check if dataset is huge, but here we can try a vectorized approach on unique chars
    # Concatenating all text is slow. We'll check a sample of unique tokens.
    sample_tokens = df["before"].sample(n=min(10000, len(df)), random_state=42).tolist()
    all_chars = set("".join(str(t) for t in sample_tokens))
    has_digits = any(c.isdigit() for c in all_chars)
    has_latin = any("a" <= c.lower() <= "z" for c in all_chars)
    has_cyrillic = any(
        "\u0400" <= c <= "\u04ff" for c in all_chars
    )  # Basic Cyrillic block

    print("\nCharacter Set Composition (Sampled):")
    print(f"  Contains Digits: {has_digits}")
    print(f"  Contains Latin: {has_latin}")
    print(f"  Contains Cyrillic: {has_cyrillic}")

    # --- 3. Feature/Signal Relationships (Unstructured/Meta-Features) ---
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Relationship between Class and Normalization Rate
    print("Normalization Rate by Class (Top 10):")
    # We group by class and calculate the mean of 'is_changed'
    class_change_stats = (
        df.groupby("class")["is_changed"]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
        .head(10)
    )

    for cls in class_change_stats.index:
        rate = class_change_stats.loc[cls, "mean"]
        print(
            f"  {cls:<15} : {rate:.4f} (proportion of tokens in this class that change)"
        )

    # Relationship between Input Length and Class
    print("\nAverage Input Token Length by Class (Top 10):")
    class_len_stats = (
        df.groupby("class")["len_before"].mean().loc[class_change_stats.index]
    )
    for cls, avg_len in class_len_stats.items():
        print(f"  {cls:<15} : {avg_len:.4f} chars")

    # Relationship between Input Length and Change Probability
    # Bin lengths to see if longer/shorter tokens are more likely to change
    # We use qcut for roughly equal sized bins, or cut for fixed ranges.
    # Given the discrete nature of token lengths (often small integers), simple grouping is better.
    print("\nChange Probability by Input Length (First 10 lengths):")
    len_change_stats = df.groupby("len_before")["is_changed"].mean().head(10)
    for length, rate in len_change_stats.items():
        print(f"  Length {length:<2} : {rate:.4f}")

    # Sentence Context Analysis
    # Group by sentence_id to see sentence length distribution
    sent_lengths = df.groupby("sentence_id").size()
    print("\nSentence Length Statistics (Tokens per Sentence):")
    print(f"  Mean: {sent_lengths.mean():.4f}")
    print(f"  Min : {sent_lengths.min():.4f}")
    print(f"  Max : {sent_lengths.max():.4f}")


if __name__ == "__main__":
    main()
