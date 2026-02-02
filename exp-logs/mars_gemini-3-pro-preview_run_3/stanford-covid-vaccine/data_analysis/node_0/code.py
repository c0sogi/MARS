import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    SEED = 42
    np.random.seed(SEED)

    METADATA_PATH = "./metadata/train.parquet"

    if not os.path.exists(METADATA_PATH):
        print(f"Error: {METADATA_PATH} not found.")
        return

    # Load data
    df = pd.read_parquet(METADATA_PATH)

    # 2. Target Variable Analysis
    print("==== TARGET VARIABLE ANALYSIS ====")
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Flatten lists to analyze global distribution
    # Note: Targets are lists of length seq_scored (68).
    exploded_targets = df[target_cols].apply(lambda x: x.explode()).astype(float)

    print(f"{'Target':<15} {'Mean':<10} {'Std':<10} {'Skew':<10} {'Kurtosis':<10}")
    print("-" * 60)

    for col in target_cols:
        series = exploded_targets[col].dropna()
        t_mean = series.mean()
        t_std = series.std()
        t_skew = skew(series)
        t_kurt = kurtosis(series)

        print(
            f"{col:<15} {t_mean:<10.4f} {t_std:<10.4f} {t_skew:<10.4f} {t_kurt:<10.4f}"
        )

    print("\nTarget Correlation Matrix (Pearson):")
    print(exploded_targets.corr().round(4))

    # 3. Input Data Analysis (Modality: Text/Sequence)
    print("\n==== INPUT DATA ANALYSIS (SEQUENCE/TEXT) ====")

    # Length Analysis
    seq_lengths = df["sequence"].apply(len)
    print(
        f"Sequence Lengths: Mean={seq_lengths.mean():.4f}, Min={seq_lengths.min()}, Max={seq_lengths.max()}"
    )

    # Vocabulary Analysis
    def report_vocab(series, name):
        all_text = "".join(series.tolist())
        unique, counts = np.unique(list(all_text), return_counts=True)
        total = counts.sum()
        dist = {k: v / total for k, v in zip(unique, counts)}
        print(f"\n{name} Character Distribution:")
        for char, freq in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  '{char}': {freq:.4f}")

    report_vocab(df["sequence"], "Sequence (Nucleotides)")
    report_vocab(df["structure"], "Structure (Dot-Bracket)")
    report_vocab(df["predicted_loop_type"], "Predicted Loop Type")

    # 4. Input Data Analysis (Modality: Tabular Metadata)
    print("\n==== INPUT DATA ANALYSIS (TABULAR METADATA) ====")

    # Missing Values
    print("Missing Values per Column:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found.")
    else:
        print(missing[missing > 0])

    # Numerical/Categorical Analysis
    meta_cols = ["signal_to_noise", "SN_filter"]

    for col in meta_cols:
        if col not in df.columns:
            continue
        data = df[col]
        print(f"\nColumn: {col}")

        if data.nunique() < 10:
            print(f"  Type: Categorical")
            print(f"  Counts:\n{data.value_counts()}")
        else:
            print(f"  Type: Numerical")
            print(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}")
            print(f"  Min: {data.min():.4f}, Max: {data.max():.4f}")

            # Outliers (IQR)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
            print(f"  Outliers (IQR): {outliers} ({outliers/len(data)*100:.2f}%)")

    # 5. Feature/Signal Relationships
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # Feature Engineering for Relationship Analysis
    # We aggregate sequence properties to predict the Mean Reactivity of the sample
    feat_df = pd.DataFrame()

    # Target: Mean reactivity of the sample
    feat_df["target_mean_reactivity"] = df["reactivity"].apply(np.mean)

    # Features
    feat_df["signal_to_noise"] = df["signal_to_noise"]
    feat_df["SN_filter"] = df["SN_filter"]

    # Nucleotide content
    for char in ["A", "G", "C", "U"]:
        feat_df[f"pct_{char}"] = df["sequence"].apply(lambda s: s.count(char) / len(s))

    # Structure content (Paired vs Unpaired)
    # '.' is unpaired. '(' and ')' are paired.
    feat_df["pct_unpaired"] = df["structure"].apply(lambda s: s.count(".") / len(s))
    feat_df["pct_paired"] = 1.0 - feat_df["pct_unpaired"]

    # Correlation Analysis
    print("Correlation with Mean Reactivity:")
    corrs = feat_df.corrwith(feat_df["target_mean_reactivity"]).sort_values(
        ascending=False
    )
    print(corrs.drop("target_mean_reactivity").round(4))

    # Feature Importance (Random Forest)
    print("\nTop 5 Features (Random Forest Importance):")
    X = feat_df.drop(columns=["target_mean_reactivity"])
    y = feat_df["target_mean_reactivity"]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=6, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print(importances.head(5).round(4))

    # Redundancy Check
    print("\nRedundancy Check (Correlation > 0.90):")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column] > 0.90)]

    if high_corr:
        for col in high_corr:
            correlated_with = upper.index[upper[col] > 0.90].tolist()
            print(f"  {col} is highly correlated with {correlated_with}")
    else:
        print("  No highly correlated features found (> 0.90).")


if __name__ == "__main__":
    main()
