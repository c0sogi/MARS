import os
import random
import numpy as np
import pandas as pd
import warnings
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# 1. Setup and Configuration
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    set_seed(42)

    print("========================================")
    print("      EXPLORATORY DATA ANALYSIS         ")
    print("========================================")

    # Define paths
    TRAIN_PATH = "./metadata/train_metadata.csv"

    # Check existence
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        return

    # Load Data
    try:
        df = pd.read_csv(TRAIN_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ---------------------------------------------------------
    # 2. Target Variable Analysis
    # ---------------------------------------------------------
    print("\n[TARGET VARIABLE ANALYSIS]")

    # The target is distributed across three columns: winner_model_a, winner_model_b, winner_tie
    # We create a single categorical label for analysis
    target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]

    # Determine the primary class (argmax)
    df["target_class"] = df[target_cols].idxmax(axis=1)

    # Distribution
    class_counts = df["target_class"].value_counts()
    total_samples = len(df)

    print("Class Distribution:")
    for cls, count in class_counts.items():
        ratio = count / total_samples
        print(f"  - {cls}: {count} ({ratio:.4f})")

    # Check for Imbalance
    max_class_count = class_counts.max()
    min_class_count = class_counts.min()
    imbalance_ratio = max_class_count / min_class_count
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")
    if imbalance_ratio < 1.2:
        print("  -> The dataset is relatively balanced.")
    else:
        print("  -> The dataset shows signs of imbalance.")

    # ---------------------------------------------------------
    # 3. Input Data Analysis (Text Modality)
    # ---------------------------------------------------------
    print("\n[INPUT DATA ANALYSIS - TEXT]")

    text_cols = ["prompt", "response_a", "response_b"]

    # Check for Missing Values
    print("Missing Values per Text Column:")
    missing_stats = df[text_cols].isnull().sum()
    for col, val in missing_stats.items():
        pct = (val / total_samples) * 100
        print(f"  - {col}: {val} ({pct:.4f}%)")

    # Fill NaNs with empty string for length analysis to avoid errors
    df_text = df[text_cols].fillna("")

    # Analyze Lengths (Character and Word counts)
    stats_data = {}

    for col in text_cols:
        # Character lengths
        char_lens = df_text[col].apply(len)
        # Word lengths (simple whitespace split)
        word_lens = df_text[col].apply(lambda x: len(str(x).split()))

        stats_data[f"{col}_char_len"] = char_lens
        stats_data[f"{col}_word_len"] = word_lens

        print(f"\nStatistics for '{col}':")
        print(
            f"  - Char Length: Mean={char_lens.mean():.4f}, Std={char_lens.std():.4f}, Max={char_lens.max()}"
        )
        print(
            f"  - Word Length: Mean={word_lens.mean():.4f}, Std={word_lens.std():.4f}, Max={word_lens.max()}"
        )

    # Vocabulary Analysis (Approximation on a sample if large, but 40k is manageable)
    # We combine all text to check global vocabulary
    print("\nVocabulary Analysis:")

    # Use a simple tokenizer for estimation
    def simple_tokenize(text):
        return str(text).lower().split()

    # Sample 10% for vocabulary estimation to keep runtime low
    sample_text = df_text.sample(frac=0.1, random_state=42)
    all_tokens = []
    for col in text_cols:
        all_tokens.extend(
            [token for text in sample_text[col] for token in simple_tokenize(text)]
        )

    vocab_counter = Counter(all_tokens)
    vocab_size = len(vocab_counter)
    print(f"  - Estimated Vocabulary Size (from 10% sample): {vocab_size}")

    # Check for OOV potential (words appearing only once in the sample)
    singletons = sum(1 for count in vocab_counter.values() if count == 1)
    print(
        f"  - Rare Tokens (freq=1 in sample): {singletons} ({singletons/vocab_size:.4f} of vocab)"
    )

    # ---------------------------------------------------------
    # 4. Feature/Signal Relationships
    # ---------------------------------------------------------
    print("\n[FEATURE/SIGNAL RELATIONSHIPS]")

    # Create Meta-Features for Structured Analysis
    # We use the lengths calculated earlier
    meta_df = pd.DataFrame(stats_data)

    # Add difference features
    meta_df["len_diff_char"] = (
        meta_df["response_a_char_len"] - meta_df["response_b_char_len"]
    )
    meta_df["len_diff_word"] = (
        meta_df["response_a_word_len"] - meta_df["response_b_word_len"]
    )

    # Encode target for correlation
    le = LabelEncoder()
    # Mapping: winner_model_a -> 0, winner_model_b -> 1, winner_tie -> 2 (arbitrary, just for correlation check)
    # Better to check correlation against specific outcomes.

    # Let's check correlation with "Model A Wins" probability
    # Assuming the columns in df are probabilities or 0/1
    meta_df["target_prob_a"] = df["winner_model_a"]
    meta_df["target_prob_b"] = df["winner_model_b"]
    meta_df["target_prob_tie"] = df["winner_tie"]

    # Correlation Matrix
    print("Correlations with Target (Winner Model A):")
    correlations = meta_df.corr()["target_prob_a"].sort_values(ascending=False)
    # Filter out the target columns themselves
    feature_corrs = correlations.drop(
        ["target_prob_a", "target_prob_b", "target_prob_tie"]
    )

    print(feature_corrs)

    print("\nObservation on Length Bias:")
    corr_len_diff = feature_corrs.get("len_diff_char", 0)
    print(f"  - Correlation between (Len A - Len B) and A Winning: {corr_len_diff:.4f}")
    if abs(corr_len_diff) > 0.1:
        print(
            "  -> Significant relationship: Models with longer responses tend to be favored (or penalized)."
        )
    else:
        print(
            "  -> Weak linear relationship between response length difference and winning."
        )

    # Feature Importance via Random Forest
    print("\nFeature Importance (Random Forest):")

    # Prepare X and y
    # Features: Lengths of prompt, response A, response B, and differences
    feature_cols = [c for c in meta_df.columns if "target" not in c]
    X = meta_df[feature_cols].fillna(0)
    y = df["target_class"]  # Categorical target

    # Train lightweight RF
    clf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    clf.fit(X, y)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Meta-Features predicting the winner:")
    for i in range(min(5, len(feature_cols))):
        idx = indices[i]
        print(f"  {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")

    # ---------------------------------------------------------
    # 5. Metadata Analysis (Model Identity)
    # ---------------------------------------------------------
    print("\n[METADATA ANALYSIS - MODEL IDENTITY]")
    # The columns model_a and model_b exist in train
    if "model_a" in df.columns and "model_b" in df.columns:
        unique_models = pd.concat([df["model_a"], df["model_b"]]).unique()
        print(f"Number of unique models in arena: {len(unique_models)}")

        # Check cardinality
        if len(unique_models) > 50:
            print(
                f"  -> High cardinality categorical feature ({len(unique_models)} categories)."
            )

        # Simple win rate analysis for top frequent models
        # We look at cases where model is in position A
        model_counts = df["model_a"].value_counts().head(5)
        print("\nWin rates for top 5 most frequent models (when in position A):")
        for model_name in model_counts.index:
            subset = df[df["model_a"] == model_name]
            win_rate = subset["winner_model_a"].mean()
            count = len(subset)
            print(f"  - {model_name}: Win Rate={win_rate:.4f} (n={count})")
    else:
        print("Model identity columns not found.")

    print("\n========================================")
    print("           EDA COMPLETE                 ")
    print("========================================")


if __name__ == "__main__":
    main()
