import pandas as pd
import numpy as np
import os
import random
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def print_header(title):
    print(f"\n{'='*10} {title} {'='*10}")


def run_eda():
    # --- 1. Load Data ---
    data_path = "./metadata/train.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    # --- 2. Target Variable Analysis ---
    print_header("TARGET VARIABLE ANALYSIS")

    # Determine the target class from one-hot columns
    # 0: model_a, 1: model_b, 2: tie
    def get_target_label(row):
        if row["winner_model_a"] == 1:
            return "model_a"
        elif row["winner_model_b"] == 1:
            return "model_b"
        elif row["winner_tie"] == 1:
            return "tie"
        else:
            return "unknown"

    df["target_class"] = df.apply(get_target_label, axis=1)

    # Filter out any potential unknown rows (though metadata generation should have handled this)
    df = df[df["target_class"] != "unknown"].copy()

    total_samples = len(df)
    class_counts = df["target_class"].value_counts()

    print(f"Total Samples: {total_samples}")
    print("\nClass Distribution:")
    for cls, count in class_counts.items():
        ratio = count / total_samples
        print(f"  {cls.ljust(10)}: {count} ({ratio:.4f})")

    # --- 3. Input Data Analysis (Text Modality) ---
    print_header("INPUT DATA ANALYSIS (TEXT)")

    text_cols = ["prompt", "response_a", "response_b"]

    # Helper to calculate stats
    def get_text_stats(series):
        # Fill NaNs with empty string just in case
        series = series.fillna("")
        char_lens = series.str.len()
        word_lens = series.str.split().str.len()
        return char_lens, word_lens

    vocab_counter = Counter()

    for col in text_cols:
        print(f"\nAnalysis for column: '{col}'")
        char_lens, word_lens = get_text_stats(df[col])

        print(
            f"  Character Lengths: Mean={char_lens.mean():.4f}, Std={char_lens.std():.4f}, Min={char_lens.min()}, Max={char_lens.max()}"
        )
        print(
            f"  Word Counts:       Mean={word_lens.mean():.4f}, Std={word_lens.std():.4f}, Min={word_lens.min()}, Max={word_lens.max()}"
        )

        # Update global vocabulary (simple whitespace split)
        # We process in chunks to avoid memory issues if dataset is huge, though 40k is manageable
        # Using a simple set update for speed
        tokens = set()
        df[col].fillna("").str.split().apply(
            lambda x: tokens.update(x) if isinstance(x, list) else None
        )
        vocab_counter.update(tokens)

    print(
        f"\nGlobal Vocabulary Estimate (Unique whitespace-separated tokens across all text cols): {len(vocab_counter)}"
    )

    # --- 4. Input Data Analysis (Categorical Modality) ---
    # Although the main input is text, model_a and model_b are categorical inputs provided in train
    print_header("INPUT DATA ANALYSIS (CATEGORICAL)")

    cat_cols = ["model_a", "model_b"]
    for col in cat_cols:
        unique_vals = df[col].nunique()
        counts = df[col].value_counts()
        rare_threshold = 0.01 * len(df)
        rare_count = (counts < rare_threshold).sum()

        print(f"\nColumn: {col}")
        print(f"  Cardinality: {unique_vals}")
        print(f"  Rare Categories (<1% freq): {rare_count}")
        if unique_vals < 20:
            print(f"  Values: {counts.index.tolist()}")
        else:
            print(f"  Top 5 Values: {counts.head(5).index.tolist()}")

    # --- 5. Feature/Signal Relationships ---
    print_header("FEATURE/SIGNAL RELATIONSHIPS")

    # 5a. Unstructured Relationships (Meta-features)
    # Create length features
    df["len_a"] = df["response_a"].fillna("").str.len()
    df["len_b"] = df["response_b"].fillna("").str.len()
    df["len_diff"] = df["len_a"] - df["len_b"]

    # Encode target for correlation: 0: Tie, 1: A, 2: B (Arbitrary mapping for correlation check)
    # Better mapping for "Length vs Winner":
    # Let's check correlation between (len_a - len_b) and (winner_a - winner_b)
    # winner_val: 1 if A wins, -1 if B wins, 0 if Tie
    def get_winner_score(row):
        if row["winner_model_a"] == 1:
            return 1
        if row["winner_model_b"] == 1:
            return -1
        return 0

    df["winner_score"] = df.apply(get_winner_score, axis=1)

    corr = df["len_diff"].corr(df["winner_score"])
    print("\nRelationship: Response Length Difference vs. Winner Score (1=A, -1=B)")
    print(f"  Correlation (Pearson): {corr:.4f}")
    print(
        "  (Positive correlation implies longer responses tend to win if A is longer than B)"
    )

    # 5b. Feature Importance (Structured)
    print("\nFeature Importance (Random Forest on Meta-Features):")

    # Prepare Meta-Features
    # 1. Lengths of prompt, res_a, res_b
    # 2. Model IDs (Label Encoded)

    df["len_prompt"] = df["prompt"].fillna("").str.len()

    le = LabelEncoder()
    # Fit on both columns to ensure consistent encoding
    all_models = pd.concat([df["model_a"], df["model_b"]]).unique()
    le.fit(all_models)

    df["model_a_enc"] = le.transform(df["model_a"])
    df["model_b_enc"] = le.transform(df["model_b"])

    features = [
        "len_prompt",
        "len_a",
        "len_b",
        "len_diff",
        "model_a_enc",
        "model_b_enc",
    ]
    X = df[features]
    y = df["target_class"]  # Multi-class target

    # Train lightweight RF
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("  Top Features:")
    for i in range(min(5, len(features))):
        feat_name = features[indices[i]]
        score = importances[indices[i]]
        print(f"    {i+1}. {feat_name.ljust(15)}: {score:.4f}")

    # Check for redundancy (Correlation > 0.90 among numerical features)
    print("\nRedundancy Check (Correlation > 0.90):")
    num_features = ["len_prompt", "len_a", "len_b", "len_diff"]
    corr_matrix = df[num_features].corr().abs()
    high_corr_pairs = []

    # Iterate over upper triangle
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.90:
                high_corr_pairs.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    if high_corr_pairs:
        for f1, f2, val in high_corr_pairs:
            print(f"  {f1} - {f2}: {val:.4f}")
    else:
        print("  No highly collinear pairs found among length features.")


if __name__ == "__main__":
    run_eda()
