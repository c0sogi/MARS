import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
METADATA_TRAIN_PATH = "./metadata/train.csv"
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def analyze_targets(df):
    print("==== TARGET VARIABLE ANALYSIS ====")
    target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]

    # Check for missing values in targets
    nulls = df[target_cols].isnull().sum().sum()
    if nulls > 0:
        print(f"Warning: {nulls} missing values found in target columns.")

    # Determine hard labels for classification balance analysis
    # We use idxmax to get the column name, then map to a simpler label
    df["winner_class"] = df[target_cols].idxmax(axis=1)

    # Class Balance
    class_counts = df["winner_class"].value_counts(normalize=True)
    print("Class Balance Ratios:")
    for label, ratio in class_counts.items():
        print(f"{label:<20}: {ratio:.4f}")

    # Descriptive stats for the probabilities
    print("\nTarget Probability Distributions (Mean/Std):")
    stats = df[target_cols].agg(["mean", "std"])
    for col in target_cols:
        print(
            f"{col:<20}: Mean={stats.loc['mean', col]:.4f}, Std={stats.loc['std', col]:.4f}"
        )


def analyze_text_column(df, col_name):
    # Fill NaNs with empty string for analysis
    series = df[col_name].fillna("")

    # Lengths
    char_lens = series.str.len()
    word_lens = series.str.split().str.len()

    print(f"\n--- {col_name} Analysis ---")
    print(
        f"Character Lengths: Mean={char_lens.mean():.4f}, Std={char_lens.std():.4f}, Max={char_lens.max()}"
    )
    print(
        f"Word Counts:       Mean={word_lens.mean():.4f}, Std={word_lens.std():.4f}, Max={word_lens.max()}"
    )

    return char_lens, word_lens


def analyze_text_data(df):
    print("\n==== INPUT DATA ANALYSIS (TEXT) ====")

    text_cols = ["prompt", "response_a", "response_b"]

    # Store lengths for feature importance later
    meta_features = pd.DataFrame(index=df.index)

    for col in text_cols:
        c_len, w_len = analyze_text_column(df, col)
        meta_features[f"{col}_char_len"] = c_len
        meta_features[f"{col}_word_len"] = w_len

    # Vocabulary Size Estimation
    # We use a subset if data is huge, but 40k is manageable for a quick fit
    print("\nVocabulary Analysis:")
    try:
        # Combine all text to build a shared vocabulary estimate
        # We limit features to avoid OOM on very large vocabs during simple EDA
        vectorizer = CountVectorizer(max_features=100000, stop_words="english")

        # Sample 10% for speed if needed, but here we try full fit on combined text
        # Concatenate a sample to estimate vocab
        sample_text = pd.concat(
            [df[c].fillna("").sample(frac=0.5, random_state=SEED) for c in text_cols]
        )
        vectorizer.fit(sample_text)
        vocab_size = len(vectorizer.vocabulary_)
        print(f"Estimated Unique Vocabulary Size (Top 50% sample): {vocab_size}")

    except Exception as e:
        print(f"Vocabulary analysis failed: {e}")

    return meta_features


def analyze_relationships(df, meta_features):
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # 1. Categorical Analysis (Model Names)
    print("Categorical Analysis (Model Identities):")
    model_cols = ["model_a", "model_b"]
    for col in model_cols:
        if col in df.columns:
            unique_vals = df[col].nunique()
            print(f"{col}: {unique_vals} unique models.")
            if unique_vals > 50:
                print(f"   -> High cardinality detected for {col}.")

    # 2. Construct Analysis DataFrame
    # Combine meta features with targets and model identities
    analysis_df = meta_features.copy()

    # Add derived features
    analysis_df["len_diff_char"] = (
        analysis_df["response_a_char_len"] - analysis_df["response_b_char_len"]
    )
    analysis_df["len_diff_word"] = (
        analysis_df["response_a_word_len"] - analysis_df["response_b_word_len"]
    )

    # Encode Model Names if present
    le = LabelEncoder()
    for col in model_cols:
        if col in df.columns:
            # Handle potential NaNs in model names by treating as 'unknown'
            cleaned_col = df[col].fillna("unknown").astype(str)
            analysis_df[f"{col}_encoded"] = le.fit_transform(cleaned_col)

    # Add targets
    target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    for t in target_cols:
        analysis_df[t] = df[t]

    # 3. Correlation Analysis
    # Check correlation between length difference and Model A winning
    # If Model A is much longer (positive diff), does it win more?
    corr_char = analysis_df["len_diff_char"].corr(analysis_df["winner_model_a"])
    print(
        f"\nCorrelation (Response A Length - Response B Length vs Winner A): {corr_char:.4f}"
    )

    # 4. Feature Importance (Lightweight RF)
    print("\nFeature Importance (Random Forest):")

    # Prepare X and y
    # Features: Lengths, diffs, encoded model names
    feature_cols = [c for c in analysis_df.columns if c not in target_cols]
    X = analysis_df[feature_cols].fillna(0)

    # Target: Hard label (0, 1, 2)
    # We map the probability distribution to a single class for RF importance check
    y_labels = df[target_cols].idxmax(axis=1)
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_labels)

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    # Extract importances
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )

    print("Top 5 Features predicting the winner:")
    for feat, imp in importances.head(5).items():
        print(f"{feat:<25}: {imp:.4f}")


def main():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(METADATA_TRAIN_PATH):
        print(f"Error: {METADATA_TRAIN_PATH} not found.")
        return

    df = pd.read_csv(METADATA_TRAIN_PATH)

    # 2. Target Analysis
    analyze_targets(df)

    # 3. Text Analysis
    meta_features = analyze_text_data(df)

    # 4. Feature Relationships
    analyze_relationships(df, meta_features)


if __name__ == "__main__":
    main()
