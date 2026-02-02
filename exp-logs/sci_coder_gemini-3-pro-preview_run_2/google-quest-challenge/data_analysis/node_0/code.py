import pandas as pd
import numpy as np
import os
import random
import warnings
from scipy.stats import skew, kurtosis, spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# ==========================================
# Configuration & Setup
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Suppress warnings for clean output
    warnings.filterwarnings("ignore")
    set_seed(42)

    # ==========================================
    # 1. Data Integrity & Loading
    # ==========================================
    # Using the pre-split training metadata to prevent leakage
    data_path = "./metadata/train.csv"
    df = pd.read_csv(data_path)

    # Define Target Columns
    target_cols = [
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

    # Verify targets exist
    available_targets = [c for c in target_cols if c in df.columns]
    if len(available_targets) != 30:
        print(f"Warning: Expected 30 targets, found {len(available_targets)}")

    # ==========================================
    # 2. Target Variable Analysis
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")

    # Calculate stats across all target columns
    target_df = df[available_targets]

    # Distribution stats
    means = target_df.mean()
    stds = target_df.std()

    # Normality checks (Skew & Kurtosis)
    # Skew: 0 = normal, >0 = right tail, <0 = left tail
    # Kurtosis: 3 = normal (Fisher definition subtracts 3, so 0 is normal). Scipy defaults to Fisher.
    skews = target_df.apply(skew)
    kurtoses = target_df.apply(kurtosis)

    print(f"Number of Target Variables: {len(available_targets)}")
    print(f"Global Target Mean: {means.mean():.4f}")
    print(f"Global Target Std Dev: {stds.mean():.4f}")

    print("\n-- Normality Check (Skewness & Kurtosis) --")
    print(
        f"Average Skewness: {skews.mean():.4f} (Min: {skews.min():.4f}, Max: {skews.max():.4f})"
    )
    print(
        f"Average Kurtosis: {kurtoses.mean():.4f} (Min: {kurtoses.min():.4f}, Max: {kurtoses.max():.4f})"
    )

    # Identify most skewed target
    most_skewed_col = skews.abs().idxmax()
    print(
        f"Most Skewed Target: '{most_skewed_col}' (Skew: {skews[most_skewed_col]:.4f})"
    )

    # ==========================================
    # 3. Input Data Analysis (Text Modality)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (TEXT)")

    text_cols = ["question_title", "question_body", "answer"]
    # Ensure text columns are strings
    for col in text_cols:
        df[col] = df[col].astype(str).fillna("")

    # Length Analysis
    print("-- Sequence Lengths --")
    for col in text_cols:
        # Character counts
        char_lens = df[col].apply(len)
        # Word counts (simple whitespace split)
        word_lens = df[col].apply(lambda x: len(x.split()))

        print(f"Feature: {col}")
        print(
            f"  Char Count -> Mean: {char_lens.mean():.4f}, Std: {char_lens.std():.4f}, Max: {char_lens.max()}"
        )
        print(
            f"  Word Count -> Mean: {word_lens.mean():.4f}, Std: {word_lens.std():.4f}, Max: {word_lens.max()}"
        )

    # Vocabulary Analysis
    print("\n-- Vocabulary Analysis --")
    # Combine all text to build a shared vocabulary estimate
    all_text = pd.concat([df[c] for c in text_cols])

    # Use CountVectorizer with basic settings to estimate vocab size
    # Limiting max_features to avoid OOM on very large datasets, though 4k rows is small.
    vectorizer = CountVectorizer(min_df=2, max_features=None)
    vectorizer.fit(all_text)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"Estimated Unique Vocabulary Size (min_df=2): {vocab_size}")

    # ==========================================
    # 4. Input Data Analysis (Categorical)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (CATEGORICAL)")

    cat_cols = ["category", "host"]
    valid_cat_cols = [c for c in cat_cols if c in df.columns]

    for col in valid_cat_cols:
        unique_vals = df[col].nunique()
        print(f"Column: {col}")
        print(f"  Cardinality: {unique_vals}")

        # Check for rare labels (< 1%)
        counts = df[col].value_counts(normalize=True)
        rare_labels = counts[counts < 0.01]
        if not rare_labels.empty:
            print(f"  Rare Labels (<1% freq): {len(rare_labels)} categories found.")
        else:
            print(f"  Rare Labels: None.")

    # ==========================================
    # 5. Feature/Signal Relationships
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Strategy:
    # 1. Create meta-features (lengths).
    # 2. Encode categorical features.
    # 3. Compute correlation with a proxy target (Mean of all targets).
    # 4. Train a lightweight RF to find importance of meta-features vs targets.

    # Prepare Tabular Representation
    analysis_df = pd.DataFrame()

    # Meta-features
    for col in text_cols:
        analysis_df[f"{col}_len_char"] = df[col].apply(len)
        analysis_df[f"{col}_len_word"] = df[col].apply(lambda x: len(x.split()))

    # Categorical features
    for col in valid_cat_cols:
        le = LabelEncoder()
        analysis_df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    # Proxy Target: Mean of all 30 labels (represents general "quality/intensity")
    # Alternatively, we could analyze specific targets, but aggregate is good for general EDA.
    target_proxy = target_df.mean(axis=1)

    # Correlation Analysis
    print("-- Meta-Feature Correlations (Spearman with Mean Target) --")
    correlations = {}
    for col in analysis_df.columns:
        corr, _ = spearmanr(analysis_df[col], target_proxy)
        correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # Redundancy Check (Collinearity)
    print("\n-- Redundancy Check (Collinear Pairs > 0.90) --")
    corr_matrix = analysis_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    found_collinear = False
    for col in to_drop:
        # Find the feature it correlates with
        correlated_feats = upper.index[upper[col] > 0.90].tolist()
        for feat in correlated_feats:
            print(f"  {col} <--> {feat}")
            found_collinear = True

    if not found_collinear:
        print("  No highly collinear pairs found.")

    # Feature Importance (Lightweight Random Forest)
    print("\n-- Feature Importance (Random Forest Regressor) --")
    rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(analysis_df, target_proxy)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Features predicting Mean Target:")
    for i in range(min(5, len(indices))):
        feat_name = analysis_df.columns[indices[i]]
        score = importances[indices[i]]
        print(f"  {i+1}. {feat_name}: {score:.4f}")


if __name__ == "__main__":
    main()
