import pandas as pd
import numpy as np
import os
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


def analyze_text_data(df):
    print("INPUT DATA ANALYSIS")

    # Feature Engineering for Analysis
    df["text_len_char"] = df["text"].astype(str).apply(len)
    df["text_len_word"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    df["selected_len_char"] = df["selected_text"].astype(str).apply(len)
    df["selected_len_word"] = (
        df["selected_text"].astype(str).apply(lambda x: len(x.split()))
    )

    # 1. Lengths
    print("--- Sequence Lengths (Full Text) ---")
    stats = df["text_len_char"].describe()
    print(
        f"Character Count - Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, Min: {stats['min']:.4f}, Max: {stats['max']:.4f}"
    )

    stats_word = df["text_len_word"].describe()
    print(
        f"Word Count      - Mean: {stats_word['mean']:.4f}, Std: {stats_word['std']:.4f}, Min: {stats_word['min']:.4f}, Max: {stats_word['max']:.4f}"
    )

    print("\n--- Sequence Lengths (Selected Text) ---")
    stats_sel = df["selected_len_char"].describe()
    print(
        f"Character Count - Mean: {stats_sel['mean']:.4f}, Std: {stats_sel['std']:.4f}, Min: {stats_sel['min']:.4f}, Max: {stats_sel['max']:.4f}"
    )

    # 2. Vocabulary
    print("\n--- Vocabulary Analysis ---")
    # Simple whitespace tokenization for speed and robustness
    all_words = " ".join(df["text"].astype(str)).lower().split()
    unique_words = set(all_words)
    vocab_size = len(unique_words)
    total_words = len(all_words)

    print(f"Total Words: {total_words}")
    print(f"Unique Vocabulary Size: {vocab_size}")
    print(f"Lexical Diversity: {(vocab_size/total_words):.4f}")

    return df


def analyze_relationships(df):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Calculate Meta-Features
    # 1. Jaccard Similarity between Text and Selected Text
    df["jaccard_score"] = df.apply(
        lambda x: jaccard(x["text"], x["selected_text"]), axis=1
    )

    # 2. Length Ratio
    # Avoid division by zero
    df["len_ratio"] = df["selected_len_char"] / df["text_len_char"].replace(0, 1)

    print("--- Meta-Feature Relationships (Grouped by Sentiment) ---")

    # Group by Sentiment
    sentiment_groups = df.groupby("sentiment")

    for name, group in sentiment_groups:
        print(f"\nSentiment: {name.upper()}")
        print(f"  Count: {len(group)}")
        print(
            f"  Avg Jaccard Score (Text vs Selected): {group['jaccard_score'].mean():.4f}"
        )
        print(
            f"  Avg Length Ratio (Selected / Text):   {group['len_ratio'].mean():.4f}"
        )
        print(
            f"  Avg Full Text Length (chars):         {group['text_len_char'].mean():.4f}"
        )
        print(
            f"  Avg Selected Text Length (chars):     {group['selected_len_char'].mean():.4f}"
        )

        # Check for high overlap
        perfect_matches = (group["jaccard_score"] > 0.95).sum()
        print(
            f"  Exact Match Ratio (Jaccard > 0.95):   {(perfect_matches / len(group)):.4f}"
        )

    print("\n--- Correlation Analysis ---")
    # Correlation between lengths
    corr_len = df[["text_len_char", "selected_len_char"]].corr().iloc[0, 1]
    print(f"Pearson Correlation (Text Length vs Selected Length): {corr_len:.4f}")

    # Correlation between text length and sentiment (encoding sentiment as ordinal for rough check: neg=-1, neu=0, pos=1)
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    df["sentiment_encoded"] = df["sentiment"].map(sentiment_map)
    corr_sent_len = df[["sentiment_encoded", "text_len_char"]].corr().iloc[0, 1]
    print(f"Pearson Correlation (Sentiment vs Text Length): {corr_sent_len:.4f}")


def main():
    set_seed(42)

    # 1. Data Integrity
    print("DATA INTEGRITY")
    data_path = "./metadata/train_metadata.csv"
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    # Ensure strict usage of training set
    print(f"Source: {data_path}")
    print(f"Shape: {df.shape}")

    # Check for missing values in critical columns
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing Values Detected:")
        print(missing[missing > 0])
        # Drop for analysis
        df = df.dropna(subset=["text", "selected_text", "sentiment"])
    else:
        print("No missing values in critical columns.")
    print("-" * 30)
    print()

    # 2. Target Variable Analysis
    # In this extraction task, 'sentiment' is the conditioning class,
    # but 'selected_text' is the extraction target. We analyze sentiment distribution here.
    print("TARGET VARIABLE ANALYSIS")
    print("--- Sentiment Distribution ---")
    counts = df["sentiment"].value_counts()
    ratios = df["sentiment"].value_counts(normalize=True)

    for label in counts.index:
        print(
            f"{label.ljust(10)}: Count = {counts[label]}, Ratio = {ratios[label]:.4f}"
        )

    # Calculate imbalance
    max_ratio = ratios.max()
    min_ratio = ratios.min()
    print(f"\nClass Imbalance Ratio (Max/Min): {(max_ratio/min_ratio):.4f}")
    print("-" * 30)
    print()

    # 3. Input Data Analysis (Text Modality)
    df = analyze_text_data(df)
    print("-" * 30)
    print()

    # 4. Feature/Signal Relationships
    analyze_relationships(df)
    print("-" * 30)


if __name__ == "__main__":
    main()
