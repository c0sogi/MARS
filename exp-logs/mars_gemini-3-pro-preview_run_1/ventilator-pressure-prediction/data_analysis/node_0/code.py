import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import random

# Configuration
DATA_PATH = "./metadata/train.csv"
SAMPLE_SIZE_RF = 100000  # Number of rows to sample for Random Forest to ensure speed
RANDOM_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(RANDOM_SEED)

    print("=== DATA LOADING ===")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")

    # Define column types based on dataset description
    # id and breath_id are identifiers
    target_col = "pressure"
    numerical_cols = ["time_step", "u_in"]
    categorical_cols = [
        "R",
        "C",
        "u_out",
    ]  # R and C are discrete physical settings, u_out is binary

    # 1. Data Integrity
    # (Performed implicitly by using the metadata split which guarantees no leakage)

    print("\n=== TARGET VARIABLE ANALYSIS ===")
    target = df[target_col]

    # Distribution stats
    mean_val = target.mean()
    std_val = target.std()
    min_val = target.min()
    max_val = target.max()

    # Normality check
    skewness = target.skew()
    kurtosis = target.kurt()

    print(f"Target: {target_col}")
    print(f"Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"Min: {min_val:.4f}, Max: {max_val:.4f}")
    print(f"Skewness: {skewness:.4f} (Normal=0)")
    print(f"Kurtosis: {kurtosis:.4f} (Normal=3)")

    print("\n=== INPUT DATA ANALYSIS (TABULAR) ===")

    # Missing Values
    print("--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        for col in df.columns:
            if missing[col] > 0:
                print(f"{col}: {missing[col]} missing ({missing_pct[col]:.4f}%)")

    # Numerical Analysis
    print("\n--- Numerical Features ---")
    for col in numerical_cols:
        col_data = df[col]
        mean_c = col_data.mean()
        std_c = col_data.std()
        min_c = col_data.min()
        max_c = col_data.max()

        # Outliers using IQR
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

        print(f"Feature: {col}")
        print(
            f"  Stats: Mean={mean_c:.4f}, Std={std_c:.4f}, Min={min_c:.4f}, Max={max_c:.4f}"
        )
        print(f"  Outliers (IQR method): {outliers} ({outliers/len(df)*100:.2f}%)")

    # Categorical Analysis
    print("\n--- Categorical/Discrete Features ---")
    for col in categorical_cols:
        counts = df[col].value_counts()
        cardinality = len(counts)
        print(f"Feature: {col}")
        print(f"  Cardinality: {cardinality}")

        # Check for rare labels (< 1%)
        total_count = len(df)
        rare_labels = [
            label for label, count in counts.items() if (count / total_count) < 0.01
        ]
        if cardinality > 50:
            print(f"  High cardinality detected (>50).")

        if rare_labels:
            print(f"  Rare labels (<1% freq): {rare_labels}")
        else:
            print(f"  No rare labels detected.")

        # Print distribution for low cardinality
        if cardinality < 10:
            dist_str = ", ".join(
                [f"{k}: {v/total_count:.2%}" for k, v in counts.items()]
            )
            print(f"  Distribution: {dist_str}")

    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # Prepare data for correlation and importance
    # We treat R, C, u_out as numerical for correlation since they are ordinal/binary physically
    analysis_features = numerical_cols + categorical_cols

    # Correlation
    print("--- Correlation Analysis ---")
    corr_matrix = df[analysis_features + [target_col]].corr(method="pearson")

    # Check correlations with target
    print(f"Correlation with {target_col}:")
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    for feat, corr in target_corr.items():
        print(f"  {feat}: {corr:.4f}")

    # Check Redundancy (Collinearity > 0.90)
    print("\n--- Redundancy Check (Correlation > 0.90) ---")
    redundant_pairs = []
    features_list = analysis_features
    for i in range(len(features_list)):
        for j in range(i + 1, len(features_list)):
            f1 = features_list[i]
            f2 = features_list[j]
            val = abs(corr_matrix.loc[f1, f2])
            if val > 0.90:
                redundant_pairs.append((f1, f2, val))

    if redundant_pairs:
        for f1, f2, val in redundant_pairs:
            print(f"  High collinearity: {f1} & {f2} (Corr: {val:.4f})")
    else:
        print("  No highly collinear feature pairs found.")

    # Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")
    # Sample data for speed
    if len(df) > SAMPLE_SIZE_RF:
        df_sample = df.sample(n=SAMPLE_SIZE_RF, random_state=RANDOM_SEED)
    else:
        df_sample = df

    X = df_sample[analysis_features]
    y = df_sample[target_col]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=RANDOM_SEED, verbose=0
    )
    rf.fit(X, y)

    importances = pd.Series(
        rf.feature_importances_, index=analysis_features
    ).sort_values(ascending=False)
    print("Top 5 Features:")
    for feat, imp in importances.head(5).items():
        print(f"  {feat}: {imp:.4f}")

    print("\n=== META-FEATURE RELATIONSHIPS (TIME SERIES) ===")
    # Analyze relationship between lung attributes (R, C) and pressure aggregates per breath
    # This helps understand if certain lung types have generally higher/lower pressures

    print("Aggregating by breath_id to analyze lung attribute effects...")
    breath_agg = df.groupby("breath_id").agg(
        {"R": "first", "C": "first", "pressure": ["mean", "max", "std"], "u_in": "mean"}
    )

    # Flatten columns
    breath_agg.columns = [
        "_".join(col).strip() if col[1] else col[0] for col in breath_agg.columns.values
    ]

    # Correlation of R and C with pressure aggregates
    meta_corr = breath_agg[
        ["R_first", "C_first", "pressure_mean", "pressure_max", "pressure_std"]
    ].corr()

    print("Correlation of Lung Attributes with Pressure Aggregates:")
    print(f"  R vs Pressure Mean: {meta_corr.loc['R_first', 'pressure_mean']:.4f}")
    print(f"  R vs Pressure Max:  {meta_corr.loc['R_first', 'pressure_max']:.4f}")
    print(f"  C vs Pressure Mean: {meta_corr.loc['C_first', 'pressure_mean']:.4f}")
    print(f"  C vs Pressure Max:  {meta_corr.loc['C_first', 'pressure_max']:.4f}")


if __name__ == "__main__":
    main()
