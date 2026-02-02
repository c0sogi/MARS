import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import re

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

DATA_PATH = "./metadata/train_metadata.csv"

# Target columns as defined in the task
TARGET_COLS = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]

TEXT_COLS = ["question_title", "question_body", "answer"]


def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title.upper()}")
    print("=" * 60)


def main():
    # --------------------------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded Training Data. Shape: {df.shape}")

    # Ensure target columns exist
    available_targets = [c for c in TARGET_COLS if c in df.columns]
    if len(available_targets) != 30:
        print(f"Warning: Expected 30 targets, found {len(available_targets)}.")

    # --------------------------------------------------------------------------
    # 2. Target Variable Analysis
    # --------------------------------------------------------------------------
    print_header("Target Variable Analysis")

    # We treat these as regression targets bounded [0,1]
    target_stats = df[available_targets].agg(["mean", "std", "min", "max"])
    skew_vals = df[available_targets].apply(skew)
    kurt_vals = df[available_targets].apply(kurtosis)

    print(f"Analyzing {len(available_targets)} Target Variables (Regression [0,1])")
    print("-" * 30)

    # Summary of stats across all targets
    print(f"Global Mean of Targets:       {target_stats.loc['mean'].mean():.4f}")
    print(f"Global Std Dev of Targets:    {target_stats.loc['std'].mean():.4f}")
    print(f"Average Skewness:             {skew_vals.mean():.4f}")
    print(f"Average Kurtosis:             {kurt_vals.mean():.4f}")

    print("\nTop 3 Most Skewed Targets:")
    print(
        skew_vals.abs()
        .sort_values(ascending=False)
        .head(3)
        .to_string(float_format="%.4f")
    )

    print("\nTop 3 Highest Mean Targets:")
    print(
        target_stats.loc["mean"]
        .sort_values(ascending=False)
        .head(3)
        .to_string(float_format="%.4f")
    )

    # --------------------------------------------------------------------------
    # 3. Input Data Analysis (Text & Categorical)
    # --------------------------------------------------------------------------
    print_header("Input Data Analysis")

    # --- Text Analysis ---
    print("--- Text Modality Analysis ---")

    vocab_counters = Counter()

    for col in TEXT_COLS:
        if col not in df.columns:
            continue

        # Fill NaNs for analysis
        series = df[col].fillna("")

        # Lengths
        char_lens = series.str.len()
        word_lens = series.str.split().str.len()

        print(f"\nFeature: {col}")
        print(
            f"  Char Length: Mean={char_lens.mean():.4f}, Std={char_lens.std():.4f}, Max={char_lens.max()}"
        )
        print(
            f"  Word Count:  Mean={word_lens.mean():.4f}, Std={word_lens.std():.4f}, Max={word_lens.max()}"
        )

        # Vocabulary Update (Simple whitespace tokenization for estimation)
        # We sample if dataset is too large to keep runtime low
        sample_text = (
            series if len(series) < 10000 else series.sample(10000, random_state=SEED)
        )
        for text in sample_text:
            vocab_counters.update(text.lower().split())

    print(
        f"\nEstimated Vocabulary Size (Unique Tokens in sample): {len(vocab_counters)}"
    )

    # --- Tabular/Categorical Analysis ---
    print("\n--- Tabular/Categorical Analysis ---")

    # Identify likely categorical columns (excluding ID, texts, targets, and original_file)
    exclude_cols = set(available_targets + TEXT_COLS + ["qa_id", "original_file"])
    cat_candidates = [
        c for c in df.columns if c not in exclude_cols and df[c].dtype == "object"
    ]

    # Common columns in this dataset are 'category', 'host'
    for col in cat_candidates:
        n_unique = df[col].nunique()
        print(f"\nColumn: {col}")
        print(f"  Cardinality: {n_unique}")

        # Check for rare labels (< 1%)
        counts = df[col].value_counts(normalize=True)
        rare_labels = counts[counts < 0.01]
        print(f"  Rare Labels (<1% freq): {len(rare_labels)} categories")
        if n_unique <= 10:
            print(f"  Distribution: {counts.to_dict()}")

    # Missing Values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("No missing values found in the training set.")
    else:
        print(missing)

    # --------------------------------------------------------------------------
    # 4. Feature/Signal Relationships
    # --------------------------------------------------------------------------
    print_header("Feature/Signal Relationships")

    # --- Structured Relationships (Meta-features vs Targets) ---
    # We will construct a meta-feature dataframe
    meta_df = pd.DataFrame()

    # Length features
    for col in TEXT_COLS:
        if col in df.columns:
            meta_df[f"{col}_len_char"] = df[col].fillna("").str.len()
            meta_df[f"{col}_len_word"] = df[col].fillna("").str.split().str.len()

    # Categorical features (Label Encoded)
    for col in cat_candidates:
        le = LabelEncoder()
        # Handle new categories in future by using simple fit_transform here for analysis
        meta_df[col] = le.fit_transform(df[col].astype(str))

    # Check correlations between meta-features and average target value
    # (A proxy for "quality" or "completeness")
    df["avg_target"] = df[available_targets].mean(axis=1)

    print("Correlation of Text Lengths with Average Target Value:")
    corrs = []
    for col in meta_df.columns:
        if "len" in col:
            corr = meta_df[col].corr(df["avg_target"], method="spearman")
            corrs.append((col, corr))

    for name, val in sorted(corrs, key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {val:.4f}")

    # --- Feature Importance (Lightweight Random Forest) ---
    print("\n--- Feature Importance (Random Forest) ---")
    print("Predicting 'avg_target' using Meta-Features (Lengths + Categories)")

    X = meta_df
    y = df["avg_target"]

    # Handle NaNs in X if any (lengths shouldn't have NaNs due to fillna above)
    X = X.fillna(0)

    # Train RF
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Meta-Features influencing Target Values:")
    for i in range(min(5, len(X.columns))):
        feat_name = X.columns[indices[i]]
        score = importances[indices[i]]
        print(f"  {i+1}. {feat_name}: {score:.4f}")

    # --- Redundancy Check ---
    print("\n--- Redundancy Check (Collinear Features > 0.90) ---")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if len(to_drop) > 0:
        print(f"Found {len(to_drop)} redundant features (Correlation > 0.90):")
        for col in to_drop:
            # Find what it correlates with
            correlated_with = upper.index[upper[col] > 0.90].tolist()
            print(f"  {col} correlates with {correlated_with}")
    else:
        print("No highly collinear meta-features found.")


if __name__ == "__main__":
    main()
