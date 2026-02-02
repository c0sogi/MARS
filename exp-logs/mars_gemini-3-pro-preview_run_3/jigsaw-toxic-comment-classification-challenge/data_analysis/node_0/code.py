import os
import sys
import numpy as np
import pandas as pd
import re
from collections import Counter
import warnings

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
ORIGINAL_TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
SEED = 42

# Label columns for this specific dataset
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def set_seed(seed):
    np.random.seed(seed)


def load_data():
    """
    Loads the training data based on the metadata split.
    Merges labels from metadata with text from the original source.
    """
    if not os.path.exists(TRAIN_META_PATH):
        print(f"Metadata file not found at {TRAIN_META_PATH}. Cannot proceed.")
        sys.exit(1)

    if not os.path.exists(ORIGINAL_TRAIN_PATH):
        print(
            f"Original train file not found at {ORIGINAL_TRAIN_PATH}. Cannot proceed."
        )
        sys.exit(1)

    # Load metadata to get the IDs and indices for the training split
    meta_df = pd.read_csv(TRAIN_META_PATH)

    # Load original text data
    # We load the full CSV, then select rows based on metadata indices to ensure alignment
    full_train_df = pd.read_csv(ORIGINAL_TRAIN_PATH)

    # Select specific rows defined in metadata
    # The metadata contains 'source_row_index' which maps to the original dataframe index
    train_indices = meta_df["source_row_index"].values
    train_df = full_train_df.iloc[train_indices].copy()

    # Ensure labels are correct (metadata is the source of truth for the split labels)
    # We drop labels from original and merge/assign from metadata to be safe,
    # though they should be identical.
    train_df = train_df.drop(columns=LABEL_COLS, errors="ignore")

    # Join labels from metadata on ID or index.
    # Since we used iloc based on source_row_index, the order is preserved if we reset index.
    # A safer merge:
    train_df = train_df.merge(meta_df[["id"] + LABEL_COLS], on="id", how="inner")

    return train_df


def analyze_targets(df):
    print("SECTION 1: TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    # 1. Class Balance Ratios
    print("Class Balance Ratios (Positive Class Frequency):")
    total_samples = len(df)
    for col in LABEL_COLS:
        pos_count = df[col].sum()
        ratio = pos_count / total_samples
        print(f"  {col:<15}: {pos_count} / {total_samples} ({ratio:.4f})")

    # 2. Multi-label analysis
    # Check how many labels a comment usually has
    label_counts = df[LABEL_COLS].sum(axis=1)
    print("\nLabel Count Distribution per Sample:")
    dist = label_counts.value_counts().sort_index()
    for count, freq in dist.items():
        pct = freq / total_samples
        print(f"  {count} labels: {freq} ({pct:.4f})")

    # 3. Correlation between labels
    print("\nLabel Correlation Matrix (Pearson):")
    corr_mat = df[LABEL_COLS].corr()
    print(corr_mat.round(4).to_string())
    print("\n")


def analyze_text_modality(df):
    print("SECTION 2: INPUT DATA ANALYSIS (TEXT)")
    print("-" * 30)

    # Handle missing text
    if df["comment_text"].isnull().any():
        n_missing = df["comment_text"].isnull().sum()
        print(
            f"Warning: Found {n_missing} missing text values. Filling with empty string."
        )
        df["comment_text"] = df["comment_text"].fillna("")

    # 1. Lengths
    # Character counts
    char_lengths = df["comment_text"].str.len()
    # Word counts (simple whitespace split)
    word_lengths = df["comment_text"].apply(lambda x: len(str(x).split()))

    print("Sequence Length Statistics:")
    print(
        f"  Character Count: Mean={char_lengths.mean():.4f}, Std={char_lengths.std():.4f}, "
        f"Min={char_lengths.min()}, Max={char_lengths.max()}"
    )
    print(
        f"  Word Count:      Mean={word_lengths.mean():.4f}, Std={word_lengths.std():.4f}, "
        f"Min={word_lengths.min()}, Max={word_lengths.max()}"
    )

    # 2. Vocabulary
    print("\nVocabulary Analysis:")
    # We'll use a simple tokenizer for EDA speed and robustness
    # Convert to lower case and find all alphanumeric sequences
    all_text = " ".join(df["comment_text"].astype(str).tolist()).lower()
    # Simple regex for tokenization
    tokens = re.findall(r"\b\w+\b", all_text)

    vocab_size = len(set(tokens))
    total_tokens = len(tokens)
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Unique Vocabulary Size: {vocab_size}")

    # Top words
    common_words = Counter(tokens).most_common(10)
    print("  Top 10 Most Common Words:")
    for word, count in common_words:
        print(f"    {word}: {count}")

    # Add meta-features to df for the next section
    df["char_length"] = char_lengths
    df["word_length"] = word_lengths
    print("\n")
    return df


def analyze_relationships(df):
    print("SECTION 3: FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    print("Unstructured (Meta-Feature) Relationships:")
    print("Correlation between Comment Length (Word Count) and Toxicity Labels:")

    # Calculate Point-Biserial Correlation (since labels are binary and length is continuous, Pearson works as an estimator)
    for col in LABEL_COLS:
        corr = df["word_length"].corr(df[col])
        print(f"  Correlation (Word Length vs {col}): {corr:.4f}")

    print("\nMean Word Count by Class:")
    for col in LABEL_COLS:
        mean_toxic = df[df[col] == 1]["word_length"].mean()
        mean_clean = df[df[col] == 0]["word_length"].mean()
        diff = mean_toxic - mean_clean
        print(
            f"  {col:<15}: Positive={mean_toxic:.4f}, Negative={mean_clean:.4f}, Diff={diff:.4f}"
        )

    # Check if 'clean' comments (no labels) have different length characteristics
    is_clean = df[LABEL_COLS].sum(axis=1) == 0
    mean_clean_all = df[is_clean]["word_length"].mean()
    mean_any_toxic = df[~is_clean]["word_length"].mean()
    print(f"\n  Clean (No Labels) Mean Length: {mean_clean_all:.4f}")
    print(f"  Any Toxic Label Mean Length:   {mean_any_toxic:.4f}")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    set_seed(SEED)

    try:
        # Load Data
        df_train = load_data()

        # Run Analysis
        analyze_targets(df_train)
        df_train = analyze_text_modality(df_train)
        analyze_relationships(df_train)

    except Exception as e:
        print(f"An error occurred during EDA: {e}")
        sys.exit(1)
