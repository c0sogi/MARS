import pandas as pd
import numpy as np
import os
import random
from scipy.stats import skew, kurtosis, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(42)

    # Define paths
    TRAIN_PATH = "./metadata/train.csv"

    # Load data
    try:
        df = pd.read_csv(TRAIN_PATH)
    except FileNotFoundError:
        print(f"Error: {TRAIN_PATH} not found.")
        return

    # Identify columns
    # Based on description, last 30 columns are targets
    # We can also identify them by name pattern or sample_submission reference,
    # but using the known list from the prompt description is safer if available.
    # Here we infer from the dataframe structure assuming the standard format.

    # All columns
    all_cols = df.columns.tolist()

    # Known text columns
    text_cols = ["question_title", "question_body", "answer"]

    # Known categorical columns (metadata)
    cat_cols = ["category", "host"]

    # Filter out known non-targets to find targets
    # qa_id, filepath, text_cols, cat_cols are features/meta
    non_target_cols = ["qa_id", "filepath"] + text_cols + cat_cols

    # In the provided dataset, targets are the specific 30 columns.
    # Let's extract them dynamically.
    # Usually targets are float and features are object/int (except qa_id).
    # A robust way given the prompt is to look for columns starting with 'question_' or 'answer_'
    # that are NOT 'question_title', 'question_body', 'answer'.

    target_cols = [
        c
        for c in all_cols
        if (c.startswith("question_") or c.startswith("answer_")) and c not in text_cols
    ]

    # Verify we have 30 targets
    if len(target_cols) != 30:
        # Fallback: assume last 30 columns are targets as per typical competition setup if names don't match
        # But based on prompt, names should match.
        pass

    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("================================")

    # ==========================================
    # 1. DATA INTEGRITY & SUMMARY
    # ==========================================
    print("\nDATA SUMMARY")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(all_cols) - len(target_cols)}")
    print(f"Number of targets: {len(target_cols)}")

    # ==========================================
    # 2. TARGET VARIABLE ANALYSIS
    # ==========================================
    print("\nTARGET VARIABLE ANALYSIS")

    # Since there are 30 targets, we summarize them.
    # We calculate stats for each, then report the distribution of those stats
    # and list a few extreme cases.

    target_stats = []
    for col in target_cols:
        vals = df[col].dropna()
        stat = {
            "column": col,
            "mean": np.mean(vals),
            "std": np.std(vals),
            "min": np.min(vals),
            "max": np.max(vals),
            "skew": skew(vals),
            "kurtosis": kurtosis(vals),
        }
        target_stats.append(stat)

    stats_df = pd.DataFrame(target_stats)

    print(f"Global Target Mean: {stats_df['mean'].mean():.4f}")
    print(f"Global Target Std:  {stats_df['std'].mean():.4f}")

    print("\nSkewness & Kurtosis Summary (averaged across 30 targets):")
    print(f"Average Skewness: {stats_df['skew'].mean():.4f}")
    print(f"Average Kurtosis: {stats_df['kurtosis'].mean():.4f}")

    print("\nTop 3 Most Skewed Targets:")
    top_skew = stats_df.sort_values(by="skew", key=abs, ascending=False).head(3)
    for _, row in top_skew.iterrows():
        print(
            f"- {row['column']}: Skew={row['skew']:.4f}, Kurtosis={row['kurtosis']:.4f}"
        )

    # ==========================================
    # 3. INPUT DATA ANALYSIS (TEXT)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (TEXT)")

    vocab_counters = {col: Counter() for col in text_cols}
    length_stats = []

    for col in text_cols:
        if col not in df.columns:
            continue

        # Fill NaNs for analysis
        series = df[col].fillna("")

        # Character lengths
        char_lens = series.str.len()

        # Word counts (simple whitespace split)
        word_lens = series.str.split().str.len()

        # Vocabulary update (using a subset for speed if needed, but dataset is small enough)
        # We'll sample if > 10k rows, but here we have ~4k rows, so full pass is fine.
        tokens = series.str.cat(sep=" ").split()
        vocab_counters[col].update(tokens)

        print(f"\nFeature: {col}")
        print(
            f"  Char Length: Mean={char_lens.mean():.4f}, Std={char_lens.std():.4f}, Max={char_lens.max()}"
        )
        print(
            f"  Word Count:  Mean={word_lens.mean():.4f}, Std={word_lens.std():.4f}, Max={word_lens.max()}"
        )
        print(f"  Vocabulary Size (approx): {len(vocab_counters[col])}")

    # ==========================================
    # 4. INPUT DATA ANALYSIS (TABULAR/METADATA)
    # ==========================================
    print("\nINPUT DATA ANALYSIS (TABULAR/METADATA)")

    # Check categorical columns
    existing_cat_cols = [c for c in cat_cols if c in df.columns]

    for col in existing_cat_cols:
        series = df[col].astype(str)
        n_unique = series.nunique()
        counts = series.value_counts(normalize=True)
        rare_count = (counts < 0.01).sum()

        print(f"\nFeature: {col}")
        print(f"  Cardinality: {n_unique}")
        print(f"  Rare Labels (<1%): {rare_count}")
        if n_unique <= 10:
            print(f"  Distribution: {counts.to_dict()}")

    # Missing values
    print("\nMissing Values Summary:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  No missing values found.")
    else:
        for col, val in missing.items():
            print(f"  {col}: {val} ({val/len(df):.2%})")

    # ==========================================
    # 5. FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print("\nFEATURE/SIGNAL RELATIONSHIPS")

    # Construct Meta-Features DataFrame
    meta_df = pd.DataFrame()

    # 1. Length features
    for col in text_cols:
        if col in df.columns:
            meta_df[f"{col}_char_len"] = df[col].fillna("").str.len()
            meta_df[f"{col}_word_count"] = df[col].fillna("").str.split().str.len()

    # 2. Categorical features (Label Encoded)
    le_dict = {}
    for col in existing_cat_cols:
        le = LabelEncoder()
        # Handle NaN as a category
        meta_df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    # Target for relationship analysis: Mean of all 30 targets (Global Quality/Intensity)
    # This simplifies the analysis to "What drives higher overall scores?"
    target_mean_vector = df[target_cols].mean(axis=1)

    # A. Correlations (Meta-features vs Target Mean)
    print("\nTop Correlations (Meta-features vs Mean Target):")
    correlations = []
    for col in meta_df.columns:
        corr, _ = spearmanr(meta_df[col], target_mean_vector)
        correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # B. Feature Importance (Random Forest)
    # Train a small RF to predict the mean target score based on meta-features
    print("\nMultivariate Feature Importance (Random Forest):")

    X = meta_df
    y = target_mean_vector

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("  Top 5 Features driving Mean Target Score:")
    for i in range(min(5, len(indices))):
        idx = indices[i]
        print(f"  {X.columns[idx]}: {importances[idx]:.4f}")

    # C. Redundancy Check (Collinearity among meta-features)
    print("\nRedundancy Check (Collinear Meta-features > 0.90):")
    corr_matrix = meta_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if not to_drop:
        print("  No highly collinear meta-features found.")
    else:
        for col in to_drop:
            # Find what it correlates with
            correlated_with = upper.index[upper[col] > 0.90].tolist()
            print(f"  {col} correlates with {correlated_with}")


if __name__ == "__main__":
    main()
