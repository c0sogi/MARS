import pandas as pd
import numpy as np
import os
import sys
import random
from collections import Counter
import re

# ---------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------
SEED = 42
TRAIN_PATH = "./metadata/train.parquet"
SAMPLE_SIZE = 1_000_000  # Sample size for efficient processing

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def analyze_target_variable(text_series):
    """
    For this task (Masked Language Modeling / Word Insertion), the 'target'
    is effectively the words themselves. We analyze the word frequency distribution.
    """
    print_section("Target Variable Analysis")

    # Tokenize a subset of words to analyze distribution
    # We use a simple whitespace split for EDA speed
    all_tokens = []
    # We iterate and extend to avoid creating one massive string
    for sentence in text_series:
        all_tokens.extend(sentence.split())

    token_counts = Counter(all_tokens)
    total_tokens = sum(token_counts.values())
    unique_tokens = len(token_counts)

    # Top words (Class Balance equivalent)
    top_n = 20
    top_tokens = token_counts.most_common(top_n)

    print(f"Total Tokens in Sample: {total_tokens}")
    print(f"Unique Tokens (Vocabulary Size): {unique_tokens}")

    print("\n--- Distribution (Top 20 Words) ---")
    print(f"{'Word':<15} {'Count':<10} {'Frequency (%)':<15}")
    for word, count in top_tokens:
        freq = (count / total_tokens) * 100
        print(f"{word:<15} {count:<10} {freq:.4f}")

    # Imbalance / Skew
    # In text, this is Zipf's law. We check how much the top 1% words cover.
    top_1_percent_count = int(unique_tokens * 0.01)
    if top_1_percent_count < 1:
        top_1_percent_count = 1

    most_common_1_percent = token_counts.most_common(top_1_percent_count)
    coverage_1_percent = sum(count for word, count in most_common_1_percent)
    coverage_ratio = (coverage_1_percent / total_tokens) * 100

    print("\n--- Imbalance/Skew Analysis ---")
    print(
        f"Top 1% of unique words cover {coverage_ratio:.4f}% of all text occurrences."
    )
    print(
        "Interpretation: Highly skewed distribution (typical of natural language, adhering to Zipf's Law)."
    )

    return token_counts


def analyze_input_data(df):
    print_section("Input Data Analysis (Text Modality)")

    # 1. Lengths Analysis
    # Character Lengths
    char_lengths = df["sentence"].str.len()
    # Word Counts
    word_counts = df["sentence"].str.split().str.len()

    def report_stats(name, series):
        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()

        # Outliers using IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_pct = (len(outliers) / len(series)) * 100

        print(f"\n--- {name} Distribution ---")
        print(f"Mean: {mean_val:.4f}")
        print(f"Std Dev: {std_val:.4f}")
        print(f"Min: {min_val:.4f}")
        print(f"Max: {max_val:.4f}")
        print(f"Outliers (IQR Method): {len(outliers)} ({outlier_pct:.4f}%)")

    report_stats("Sentence Character Length", char_lengths)
    report_stats("Sentence Word Count", word_counts)

    # 2. Vocabulary Analysis
    # (Partially covered in Target Analysis, but refining here for OOV potential)
    # We use the token_counts passed from the previous step or re-calculate if needed.
    # For efficiency, we'll assume the previous step handled the heavy lifting of tokenization.
    # Here we look at rare words.

    # Re-calculate briefly for the 'Input' section context if needed,
    # but we can reuse the logic. Let's look at 'Rare Labels' equivalent.

    # Note: We performed tokenization in analyze_target_variable.
    # To keep functions clean, we'll do a lightweight check here on the dataframe columns.

    # Check for empty strings
    empty_count = len(df[df["sentence"].str.strip() == ""])
    print(f"\n--- Data Quality ---")
    print(f"Empty/Whitespace-only sentences: {empty_count}")

    return char_lengths, word_counts


def analyze_relationships(df, char_lengths, word_counts):
    print_section("Feature/Signal Relationships")

    # 1. Structured Relationships (Correlation)
    # Correlation between Character Length and Word Count
    # This is expected to be high, but variations indicate average word length differences.
    correlation = char_lengths.corr(word_counts)
    print("\n--- Structured Relationships ---")
    print(f"Pearson Correlation (Char Length vs Word Count): {correlation:.4f}")

    # 2. Unstructured (Meta-Feature) Relationships
    # Relationship between Sentence Length and Average Word Length
    # Do longer sentences use longer words?

    # Avoid division by zero
    avg_word_lengths = char_lengths / word_counts.replace(0, 1)

    # Correlation between Word Count and Average Word Length
    meta_corr = word_counts.corr(avg_word_lengths)

    print("\n--- Unstructured (Meta-Feature) Relationships ---")
    print(
        "Hypothesis: Do longer sentences (more words) tend to use more complex (longer) words?"
    )
    print(f"Correlation (Word Count vs Avg Word Length): {meta_corr:.4f}")

    if abs(meta_corr) < 0.1:
        print(
            "Interpretation: Negligible relationship. Sentence length does not strongly dictate word complexity."
        )
    elif meta_corr > 0:
        print(
            "Interpretation: Positive relationship. Longer sentences tend to use longer words."
        )
    else:
        print(
            "Interpretation: Negative relationship. Longer sentences tend to use shorter words."
        )


def main():
    print("Starting Exploratory Data Analysis...")

    # 1. Data Integrity & Loading
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: Training metadata not found at {TRAIN_PATH}")
        return

    try:
        # Load data
        # Using pyarrow engine for parquet
        df = pd.read_parquet(TRAIN_PATH, engine="pyarrow")

        # Sampling
        if len(df) > SAMPLE_SIZE:
            print(
                f"Dataset size ({len(df)}) exceeds sample limit. Sampling {SAMPLE_SIZE} rows for EDA..."
            )
            df_sample = df.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(
                drop=True
            )
        else:
            df_sample = df.copy()

        print(f"Analysis performed on {len(df_sample)} samples.")

    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Check for missing values
    missing_vals = df_sample["sentence"].isnull().sum()
    if missing_vals > 0:
        print(
            f"Warning: Found {missing_vals} NaN values in 'sentence' column. Dropping them."
        )
        df_sample = df_sample.dropna(subset=["sentence"])

    # 2. Target Variable Analysis
    # For text generation/insertion, we analyze the vocabulary distribution
    analyze_target_variable(df_sample["sentence"])

    # 3. Input Data Analysis
    char_lengths, word_counts = analyze_input_data(df_sample)

    # 4. Feature/Signal Relationships
    analyze_relationships(df_sample, char_lengths, word_counts)

    print("\nEDA Complete.")


if __name__ == "__main__":
    main()
