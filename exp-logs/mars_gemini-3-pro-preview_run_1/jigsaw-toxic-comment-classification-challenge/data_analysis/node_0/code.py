import os
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.feature_extraction.text import CountVectorizer

# Suppress warnings and progress bars
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_training_data():
    """
    Loads the training data by merging metadata with the raw source file.
    """
    try:
        # Load metadata for the training split
        meta_path = "./metadata/train.csv"
        if not os.path.exists(meta_path):
            print(f"Metadata file not found: {meta_path}")
            return None

        meta_df = pd.read_csv(meta_path)

        # Load raw text data
        # The metadata indicates the source file is 'train.csv'
        raw_path = "./input/train.csv"
        if not os.path.exists(raw_path):
            print(f"Raw data file not found: {raw_path}")
            return None

        raw_df = pd.read_csv(raw_path)

        # Merge on ID to get text content
        # meta_df contains the labels and IDs for the specific split
        # raw_df contains IDs and text
        df = pd.merge(meta_df, raw_df[["id", "comment_text"]], on="id", how="left")

        # Handle potential missing text
        df["comment_text"] = df["comment_text"].fillna("")

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def analyze_targets(df, label_cols):
    """
    Analyzes the distribution of the target variables.
    """
    print("=" * 40)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 40)

    # 1. Class Balance
    print(f"Total Training Samples: {len(df)}")
    print("\n--- Class Balance Ratios ---")
    for col in label_cols:
        count = df[col].sum()
        ratio = count / len(df)
        print(f"{col:<15}: {count:6d} ({ratio:.4%})")

    # 2. Multi-label Analysis
    df["label_count"] = df[label_cols].sum(axis=1)
    clean_count = (df["label_count"] == 0).sum()
    clean_ratio = clean_count / len(df)

    print(f"\n--- Multi-label Statistics ---")
    print(f"Clean comments (no labels): {clean_count} ({clean_ratio:.4%})")
    print(
        f"Comments with 1+ labels   : {len(df) - clean_count} ({1 - clean_ratio:.4%})"
    )

    # Distribution of label counts
    label_counts = df["label_count"].value_counts().sort_index()
    print("\nDistribution of Label Counts per Comment:")
    for count, freq in label_counts.items():
        print(f"  {count} labels: {freq} samples")

    # 3. Correlation between labels
    print("\n--- Label Correlation (Pearson) ---")
    corr = df[label_cols].corr()
    # Print the correlation matrix in a readable format
    print(corr.round(4).to_string())


def analyze_text_modality(df, text_col="comment_text"):
    """
    Analyzes text-specific properties: lengths and vocabulary.
    """
    print("\n" + "=" * 40)
    print("INPUT DATA ANALYSIS (TEXT MODALITY)")
    print("=" * 40)

    # Calculate lengths
    # Character length
    char_lens = df[text_col].str.len()
    # Word count (simple whitespace split)
    word_lens = df[text_col].str.split().str.len()

    # 1. Length Analysis
    print("--- Sequence Lengths ---")

    stats_data = {"Char Length": char_lens, "Word Count": word_lens}

    for name, series in stats_data.items():
        print(f"\n{name} Statistics:")
        print(f"  Mean: {series.mean():.4f}")
        print(f"  Std : {series.std():.4f}")
        print(f"  Min : {series.min()}")
        print(f"  Max : {series.max()}")
        print(f"  25% : {series.quantile(0.25):.0f}")
        print(f"  50% : {series.quantile(0.50):.0f}")
        print(f"  75% : {series.quantile(0.75):.0f}")
        print(f"  99% : {series.quantile(0.99):.0f}")

    # 2. Vocabulary Analysis
    print("\n--- Vocabulary Analysis ---")
    # Use CountVectorizer to get a rough estimate of vocabulary size
    # We limit to top 100k to keep it fast, or None for full
    print("Building vocabulary (unigrams)...")
    vec = CountVectorizer(min_df=2, stop_words="english", max_features=None)
    try:
        vec.fit(df[text_col])
        vocab_size = len(vec.vocabulary_)
        print(
            f"Unique Vocabulary Size (min_df=2, english stop words removed): {vocab_size}"
        )
    except Exception as e:
        print(f"Could not compute full vocabulary: {e}")

    # Check for empty strings
    empty_count = (df[text_col].str.strip() == "").sum()
    print(f"Empty or whitespace-only strings: {empty_count}")


def analyze_relationships(df, label_cols, text_col="comment_text"):
    """
    Analyzes relationships between meta-features (length) and targets.
    """
    print("\n" + "=" * 40)
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("=" * 40)

    # Create meta-features
    df["char_len"] = df[text_col].str.len()
    df["word_count"] = df[text_col].str.split().str.len()

    print("--- Meta-Feature vs Target Relationships ---")
    print("Comparing average Word Count for Positive vs Negative classes:")

    print(
        f"{'Label':<15} | {'Avg Len (Pos)':<15} | {'Avg Len (Neg)':<15} | {'Diff':<10}"
    )
    print("-" * 65)

    for label in label_cols:
        pos_mask = df[label] == 1
        neg_mask = df[label] == 0

        if pos_mask.sum() > 0:
            avg_pos = df.loc[pos_mask, "word_count"].mean()
        else:
            avg_pos = 0.0

        if neg_mask.sum() > 0:
            avg_neg = df.loc[neg_mask, "word_count"].mean()
        else:
            avg_neg = 0.0

        diff = avg_pos - avg_neg
        print(f"{label:<15} | {avg_pos:<15.4f} | {avg_neg:<15.4f} | {diff:<10.4f}")

    # Correlation between length and toxicity
    # We create a binary 'is_toxic' if any label is present
    df["is_toxic_any"] = (df[label_cols].sum(axis=1) > 0).astype(int)

    corr_len_toxic = df["word_count"].corr(df["is_toxic_any"])
    print(f"\nCorrelation between Word Count and 'Any Toxicity': {corr_len_toxic:.4f}")

    print("\n--- Top Words per Class (Simple Frequency) ---")
    # Quick check of top words for the 'toxic' class vs 'clean' class
    # We use a simple split and counter for speed and memory efficiency on the subset

    def get_top_k_words(text_series, k=5):
        try:
            vec = CountVectorizer(stop_words="english", max_features=1000)
            X = vec.fit_transform(text_series)
            sum_words = X.sum(axis=0)
            words_freq = [
                (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
            ]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            return words_freq[:k]
        except ValueError:
            return []

    # Sample for speed if dataset is huge, but 127k is okay
    # We'll just do 'toxic' and 'clean'

    print("Top 5 words in 'toxic' comments:")
    toxic_comments = df[df["toxic"] == 1][text_col]
    if len(toxic_comments) > 0:
        top_toxic = get_top_k_words(toxic_comments)
        print(f"  {', '.join([w for w, f in top_toxic])}")
    else:
        print("  (No toxic samples found)")

    print("Top 5 words in 'clean' comments (sample of 10k):")
    clean_comments = df[df["label_count"] == 0][text_col]
    if len(clean_comments) > 0:
        # Downsample clean for speed
        clean_sample = clean_comments.sample(
            n=min(10000, len(clean_comments)), random_state=42
        )
        top_clean = get_top_k_words(clean_sample)
        print(f"  {', '.join([w for w, f in top_clean])}")
    else:
        print("  (No clean samples found)")


def main():
    set_seed(42)

    # Load Data
    print("Loading Data...")
    df = load_training_data()

    if df is None:
        print("Failed to load data. Exiting.")
        return

    label_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    # Run Analysis
    analyze_targets(df, label_cols)
    analyze_text_modality(df)
    analyze_relationships(df, label_cols)

    print("\nEDA Complete.")


if __name__ == "__main__":
    main()
