import pandas as pd
import numpy as np
import os
import re
import random
from collections import Counter
from scipy.stats import skew, kurtosis


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def perform_eda():
    seed_everything()

    # --- Configuration ---
    DATA_PATH = "./metadata/train.csv"

    # --- Load Data ---
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)

    # --- 1. DATA INTEGRITY ---
    print("DATA INTEGRITY")
    print(f"Dataset Shape: {df.shape}")

    # Check for missing values
    missing_text = df["full_text"].isna().sum()
    missing_score = df["score"].isna().sum()
    print(f"Missing 'full_text': {missing_text} ({missing_text/len(df):.4%})")
    print(f"Missing 'score': {missing_score} ({missing_score/len(df):.4%})")
    print("-" * 30)

    # --- 2. TARGET VARIABLE ANALYSIS ---
    print("\nTARGET VARIABLE ANALYSIS")
    target = df["score"]

    # Distribution
    print("Score Distribution:")
    counts = target.value_counts().sort_index()
    total = len(target)
    for score, count in counts.items():
        print(f"Score {score}: {count} ({count/total:.4f})")

    # Statistical properties (treating as numerical for regression context)
    target_mean = target.mean()
    target_std = target.std()
    target_skew = skew(target)
    target_kurt = kurtosis(target)

    print("\nTarget Statistics:")
    print(f"Mean: {target_mean:.4f}")
    print(f"Std Dev: {target_std:.4f}")
    print(f"Skewness: {target_skew:.4f}")
    print(f"Kurtosis: {target_kurt:.4f}")
    print("-" * 30)

    # --- 3. INPUT DATA ANALYSIS (TEXT MODALITY) ---
    print("\nINPUT DATA ANALYSIS (TEXT)")

    # Precompute lengths
    # Simple regex for tokenization to be robust and fast
    def get_stats(text):
        chars = len(text)
        words = len(re.findall(r"\w+", text))
        return chars, words

    # Apply to a sample if dataset is huge, but 12k is small enough to process all
    # Using numpy vectorization or list comprehension for speed
    texts = df["full_text"].astype(str).tolist()

    char_counts = []
    word_counts = []
    all_tokens = []

    for t in texts:
        c = len(t)
        tokens = re.findall(r"\w+", t.lower())
        w = len(tokens)
        char_counts.append(c)
        word_counts.append(w)
        all_tokens.extend(tokens)

    df["char_count"] = char_counts
    df["word_count"] = word_counts

    # Length Analysis
    print("Sequence Lengths (Word Count):")
    wc_series = pd.Series(word_counts)
    print(f"Mean: {wc_series.mean():.4f}")
    print(f"Std: {wc_series.std():.4f}")
    print(f"Min: {wc_series.min():.4f}")
    print(f"Max: {wc_series.max():.4f}")

    print("\nSequence Lengths (Character Count):")
    cc_series = pd.Series(char_counts)
    print(f"Mean: {cc_series.mean():.4f}")
    print(f"Std: {cc_series.std():.4f}")
    print(f"Min: {cc_series.min():.4f}")
    print(f"Max: {cc_series.max():.4f}")

    # Vocabulary Analysis
    vocab_counter = Counter(all_tokens)
    vocab_size = len(vocab_counter)
    total_tokens = len(all_tokens)

    # Check for hapax legomena (words appearing only once)
    hapax_count = sum(1 for x in vocab_counter.values() if x == 1)

    print("\nVocabulary Statistics:")
    print(f"Total Tokens: {total_tokens}")
    print(f"Unique Vocabulary Size: {vocab_size}")
    print(
        f"Percentage of Unique Words (Lexical Diversity): {(vocab_size/total_tokens):.4f}"
    )
    print(
        f"Hapax Legomena (Rare words appearing once): {hapax_count} ({hapax_count/vocab_size:.4f} of vocab)"
    )

    top_10 = vocab_counter.most_common(10)
    print(f"Top 10 most common words: {', '.join([w[0] for w in top_10])}")
    print("-" * 30)

    # --- 4. FEATURE/SIGNAL RELATIONSHIPS ---
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Correlation between Length and Target
    # Pearson (linear) and Spearman (monotonic)
    pearson_corr_word = df["word_count"].corr(df["score"], method="pearson")
    spearman_corr_word = df["word_count"].corr(df["score"], method="spearman")

    pearson_corr_char = df["char_count"].corr(df["score"], method="pearson")

    print("Correlations with Target (Score):")
    print(f"Word Count vs Score (Pearson): {pearson_corr_word:.4f}")
    print(f"Word Count vs Score (Spearman): {spearman_corr_word:.4f}")
    print(f"Char Count vs Score (Pearson): {pearson_corr_char:.4f}")

    # Meta-feature relationship: Average length per score
    print("\nAverage Word Count per Score Class:")
    avg_len_per_score = df.groupby("score")["word_count"].mean()
    for score, avg_len in avg_len_per_score.items():
        print(f"Score {score}: {avg_len:.4f} words")

    print("-" * 30)


if __name__ == "__main__":
    perform_eda()
