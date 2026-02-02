import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings

# 1. Setup and Configuration
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def analyze_targets(df):
    print_section("Target Variable Analysis")

    # The targets are stored as lists in the dataframe.
    # We need to stack them into 2D arrays (n_samples x seq_scored)
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for col in target_cols:
        # Stack lists into a numpy array
        # Note: Some rows might be filtered or have different lengths if not careful,
        # but metadata generation ensured consistency.
        data_matrix = np.vstack(df[col].values)

        # Flatten for global distribution analysis
        flat_data = data_matrix.flatten()

        mean_val = np.mean(flat_data)
        std_val = np.std(flat_data)
        min_val = np.min(flat_data)
        max_val = np.max(flat_data)
        skew = stats.skew(flat_data)
        kurt = stats.kurtosis(flat_data)

        print(f"Variable: {col}")
        print(f"  Mean: {mean_val:.4f} | Std: {std_val:.4f}")
        print(f"  Min:  {min_val:.4f} | Max: {max_val:.4f}")
        print(f"  Skew: {skew:.4f} | Kurtosis: {kurt:.4f}")
        print("-" * 30)


def analyze_inputs(df):
    print_section("Input Data Analysis")

    # 1. Sequence/Text Analysis
    print("--- Sequence & Structure (Text Modality) ---")
    text_cols = ["sequence", "structure", "predicted_loop_type"]

    for col in text_cols:
        # Length Analysis
        lengths = df[col].apply(len)
        mean_len = lengths.mean()
        max_len = lengths.max()
        min_len = lengths.min()

        # Vocabulary Analysis
        # Concatenate a subset to get unique chars efficiently
        all_text = "".join(df[col].iloc[:1000].values)
        vocab = sorted(list(set(all_text)))

        print(f"Column: {col}")
        print(
            f"  Length Distribution -> Mean: {mean_len:.1f}, Min: {min_len}, Max: {max_len}"
        )
        print(f"  Vocabulary Size: {len(vocab)}")
        print(f"  Vocabulary: {vocab}")

    # 2. Tabular Metadata Analysis
    print("\n--- Metadata (Tabular Modality) ---")
    meta_cols = ["signal_to_noise", "SN_filter", "seq_length", "seq_scored"]

    for col in meta_cols:
        if col not in df.columns:
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            mean_val = series.mean()
            std_val = series.std()
            n_missing = series.isna().sum()

            # Outliers (IQR)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()

            print(f"Feature: {col}")
            print(f"  Mean: {mean_val:.4f} | Std: {std_val:.4f}")
            print(f"  Missing: {n_missing} ({n_missing/len(df):.2%})")
            print(f"  Outliers (IQR): {outliers}")
        else:
            # Categorical
            unique_count = series.nunique()
            print(f"Feature: {col}")
            print(f"  Cardinality: {unique_count}")
            if unique_count <= 10:
                print(f"  Counts: {series.value_counts().to_dict()}")


def analyze_relationships(df):
    print_section("Feature/Signal Relationships")

    # We need to construct a tabular representation for relationship analysis.
    # We will derive features from the sequences and use the mean reactivity as the target.

    # 1. Feature Engineering
    # Count nucleotides
    df_feats = pd.DataFrame()
    df_feats["len_A"] = df["sequence"].apply(lambda x: x.count("A"))
    df_feats["len_G"] = df["sequence"].apply(lambda x: x.count("G"))
    df_feats["len_C"] = df["sequence"].apply(lambda x: x.count("C"))
    df_feats["len_U"] = df["sequence"].apply(lambda x: x.count("U"))

    # Count structure elements
    df_feats["struct_dot"] = df["structure"].apply(lambda x: x.count("."))
    df_feats["struct_open"] = df["structure"].apply(lambda x: x.count("("))
    df_feats["struct_close"] = df["structure"].apply(lambda x: x.count(")"))

    # Metadata
    if "signal_to_noise" in df.columns:
        df_feats["signal_to_noise"] = df["signal_to_noise"]
    if "SN_filter" in df.columns:
        df_feats["SN_filter"] = df["SN_filter"].astype(int)

    # Target: Mean reactivity per sample (scalar)
    # We take the mean of the 68-length vector for each sample
    y = np.array([np.mean(x) for x in df["reactivity"].values])

    # 2. Correlation Analysis
    print("--- Correlation with Mean Reactivity ---")
    # Add target to temporary df for correlation
    df_corr = df_feats.copy()
    df_corr["target_mean_reactivity"] = y

    correlations = df_corr.corr(method="pearson")["target_mean_reactivity"].drop(
        "target_mean_reactivity"
    )
    # Sort by absolute correlation
    sorted_corr = correlations.abs().sort_values(ascending=False)

    for feat, corr_val in sorted_corr.head(5).items():
        raw_corr = correlations[feat]
        print(f"  {feat}: {raw_corr:.4f}")

    # Check Redundancy (Collinearity)
    print("\n--- Feature Redundancy (Corr > 0.90) ---")
    corr_matrix = df_feats.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    if to_drop:
        for col in to_drop:
            # Find what it correlates with
            correlated_cols = upper.index[upper[col] > 0.90].tolist()
            print(f"  {col} is highly correlated with: {correlated_cols}")
    else:
        print("  No highly collinear pairs found among derived features.")

    # 3. Feature Importance (Random Forest)
    print("\n--- Random Forest Feature Importance ---")
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(df_feats, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"Target: Mean Sample Reactivity")
    for i in range(min(5, len(indices))):
        feat_name = df_feats.columns[indices[i]]
        score = importances[indices[i]]
        print(f"  {i+1}. {feat_name}: {score:.4f}")


def main():
    # Load Data
    # Using the metadata parquet file as instructed
    train_path = "./metadata/train.parquet"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    df = pd.read_parquet(train_path)

    # Run Analysis Modules
    analyze_targets(df)
    analyze_inputs(df)
    analyze_relationships(df)


if __name__ == "__main__":
    main()
