import pandas as pd
import numpy as np
import os
import random
from collections import Counter
from scipy.stats import pearsonr

# Configuration
DATA_PATH = "./metadata/train.csv"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    if (len(a) == 0) & (len(b) == 0):
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def analyze_eda():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)

    # Ensure strings
    df["text"] = df["text"].astype(str)
    df["selected_text"] = df["selected_text"].astype(str)

    print("========================================")
    print("      EXPLORATORY DATA ANALYSIS         ")
    print("========================================")
    print(f"Dataset Shape: {df.shape}")

    # 2. Target Variable Analysis (Sentiment)
    print("\n2. TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    # Distribution of Sentiment
    sentiment_counts = df["sentiment"].value_counts()
    sentiment_props = df["sentiment"].value_counts(normalize=True)

    print("Sentiment Distribution:")
    for label in sentiment_counts.index:
        count = sentiment_counts[label]
        prop = sentiment_props[label]
        print(f"  {label:<10}: {count} ({prop:.4%})")

    # Class Balance
    max_class = sentiment_counts.max()
    min_class = sentiment_counts.min()
    balance_ratio = max_class / min_class
    print(f"\nClass Balance Ratio (Max/Min): {balance_ratio:.4f}")

    # 3. Input Data Analysis (Text Modality)
    print("\n3. INPUT DATA ANALYSIS (TEXT)")
    print("-" * 30)

    # Feature Engineering for Analysis
    df["text_len_char"] = df["text"].apply(len)
    df["text_len_word"] = df["text"].apply(lambda x: len(x.split()))
    df["sel_len_char"] = df["selected_text"].apply(len)
    df["sel_len_word"] = df["selected_text"].apply(lambda x: len(x.split()))

    # Length Analysis
    print("Sequence Length Statistics (Full Text):")
    print(
        f"  Char Count - Mean: {df['text_len_char'].mean():.4f}, Std: {df['text_len_char'].std():.4f}, Max: {df['text_len_char'].max()}"
    )
    print(
        f"  Word Count - Mean: {df['text_len_word'].mean():.4f}, Std: {df['text_len_word'].std():.4f}, Max: {df['text_len_word'].max()}"
    )

    print("\nSequence Length Statistics (Selected Text):")
    print(
        f"  Char Count - Mean: {df['sel_len_char'].mean():.4f}, Std: {df['sel_len_char'].std():.4f}, Max: {df['sel_len_char'].max()}"
    )
    print(
        f"  Word Count - Mean: {df['sel_len_word'].mean():.4f}, Std: {df['sel_len_word'].std():.4f}, Max: {df['sel_len_word'].max()}"
    )

    # Vocabulary Analysis
    all_text = " ".join(df["text"].tolist())
    tokens = all_text.split()
    vocab_counter = Counter(tokens)

    vocab_size = len(vocab_counter)
    total_tokens = len(tokens)
    hapax_legomena = sum(1 for count in vocab_counter.values() if count == 1)

    print("\nVocabulary Analysis:")
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Unique Vocabulary Size: {vocab_size}")
    print(
        f"  Hapax Legomena (Rare words, freq=1): {hapax_legomena} ({hapax_legomena/vocab_size:.4%} of vocab)"
    )

    # 4. Feature/Signal Relationships
    print("\n4. FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    # Jaccard Similarity Analysis
    # This is critical for this specific task: How much of the text is selected?
    df["jaccard"] = df.apply(lambda x: jaccard(x["text"], x["selected_text"]), axis=1)

    print("Jaccard Similarity (Text vs Selected_Text) by Sentiment:")
    jaccard_stats = df.groupby("sentiment")["jaccard"].agg(["mean", "std"])
    for idx, row in jaccard_stats.iterrows():
        print(f"  {idx:<10}: Mean Jaccard = {row['mean']:.4f}, Std = {row['std']:.4f}")

    # Interpretation of Jaccard
    high_overlap = df[df["jaccard"] > 0.95]
    prop_high_overlap = len(high_overlap) / len(df)
    print(f"\nGlobal High Overlap (Jaccard > 0.95): {prop_high_overlap:.4%}")

    # Breakdown of high overlap by sentiment
    print("High Overlap Proportion by Sentiment:")
    for sentiment in df["sentiment"].unique():
        subset = df[df["sentiment"] == sentiment]
        high_overlap_subset = subset[subset["jaccard"] > 0.95]
        prop = len(high_overlap_subset) / len(subset)
        print(f"  {sentiment:<10}: {prop:.4%}")

    # Correlation Analysis
    # Do longer tweets imply longer selected text?
    corr_char, _ = pearsonr(df["text_len_char"], df["sel_len_char"])
    corr_word, _ = pearsonr(df["text_len_word"], df["sel_len_word"])

    print("\nCorrelations (Input Length vs Target Length):")
    print(f"  Character Length Correlation: {corr_char:.4f}")
    print(f"  Word Count Correlation:       {corr_word:.4f}")

    # Meta-feature relationship
    print("\nAverage Word Counts by Sentiment:")
    word_stats = df.groupby("sentiment")[["text_len_word", "sel_len_word"]].mean()
    for idx, row in word_stats.iterrows():
        diff = row["text_len_word"] - row["sel_len_word"]
        print(
            f"  {idx:<10}: Full={row['text_len_word']:.4f}, Selected={row['sel_len_word']:.4f}, Diff={diff:.4f}"
        )


if __name__ == "__main__":
    analyze_eda()
