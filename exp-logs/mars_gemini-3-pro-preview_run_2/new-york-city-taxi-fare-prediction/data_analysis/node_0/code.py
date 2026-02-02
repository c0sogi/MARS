import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_target(df, target_col):
    print("TARGET VARIABLE ANALYSIS")
    target = df[target_col]

    # Distribution stats
    print(f"Target Variable: {target_col}")
    print(f"Mean: {target.mean():.4f}")
    print(f"Std Dev: {target.std():.4f}")
    print(f"Min: {target.min():.4f}")
    print(f"Max: {target.max():.4f}")

    # Normality (Skewness and Kurtosis)
    # Using scipy.stats for efficiency on large arrays
    skewness = stats.skew(target.values, nan_policy="omit")
    kurtosis = stats.kurtosis(target.values, nan_policy="omit")

    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    print("-" * 30)


def analyze_tabular_input(df, numerical_cols):
    print("INPUT DATA ANALYSIS (TABULAR)")

    # Missing Values
    print("Missing Values per Column:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    for col in df.columns:
        print(f"{col}: {missing[col]} ({missing_pct[col]:.4f}%)")
    print()

    # Numerical Analysis
    print("Numerical Feature Statistics:")
    for col in numerical_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()

        # Outlier detection using IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()

        print(f"Feature: {col}")
        print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}")
        print(f"  Outliers (IQR method): {outliers}")
    print("-" * 30)


def analyze_relationships(df, target_col, numerical_cols):
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Sampling for expensive operations
    # Use 100,000 samples for Correlation and Random Forest to ensure speed
    SAMPLE_SIZE = min(100000, len(df))
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()

    # Preprocessing for relationships
    # Drop rows with NaNs in the sample for clean correlation/training
    df_sample = df_sample.dropna(subset=numerical_cols + [target_col])

    # 1. Correlation
    print("Correlation Analysis (Pearson):")
    corr_matrix = df_sample[numerical_cols + [target_col]].corr(method="pearson")

    # Check for redundancy (Correlation > 0.90)
    redundant_pairs = []
    # Iterate over the upper triangle
    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            col1 = numerical_cols[i]
            col2 = numerical_cols[j]
            val = corr_matrix.loc[col1, col2]
            if abs(val) > 0.90:
                redundant_pairs.append((col1, col2, val))

    if redundant_pairs:
        print("Redundant Features (Correlation > 0.90):")
        for c1, c2, val in redundant_pairs:
            print(f"  {c1} & {c2}: {val:.4f}")
    else:
        print("No highly collinear pairs found (> 0.90).")
    print()

    # 2. Feature Importance (Random Forest)
    print("Feature Importance (Random Forest):")
    X = df_sample[numerical_cols]
    y = df_sample[target_col]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42, verbose=0
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Features:")
    for i in range(min(5, len(numerical_cols))):
        idx = indices[i]
        print(f"  {numerical_cols[idx]}: {importances[idx]:.4f}")


def main():
    set_seed(42)

    # Data Loading
    data_path = "./metadata/train.parquet"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load data
    # The dataset is large (~44M rows). We load it all, but be mindful of memory.
    # 220GB RAM is sufficient for 44M rows of floats/strings.
    df = pd.read_parquet(data_path)

    # Feature Engineering for EDA
    # The dataset contains a timestamp string. We need to convert it to extract numerical features
    # for the correlation and RF analysis.
    if "pickup_datetime" in df.columns:
        # Coerce errors to NaT to handle potential garbage, though metadata check passed
        # Using format inference can be slow, but format is likely standard.
        # Given the sample "2014-03-30 12:14:00 UTC", we can try to parse efficiently.
        # To save time on 44M rows, we can infer format from the first row or just let pandas do it.
        # We will strip ' UTC' if present to speed up parsing if it's just a suffix.

        # Quick check on format
        sample_time = df["pickup_datetime"].iloc[0]
        if isinstance(sample_time, str) and sample_time.endswith(" UTC"):
            df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, -4)

        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

        # Extract features
        df["hour"] = df["pickup_datetime"].dt.hour
        df["year"] = df["pickup_datetime"].dt.year
        df["month"] = df["pickup_datetime"].dt.month
        df["day"] = df["pickup_datetime"].dt.day
        df["weekday"] = df["pickup_datetime"].dt.dayofweek

    # Define Column Groups
    target_col = "fare_amount"

    # Identify numerical columns automatically, but exclude target and key
    exclude_cols = {"key", target_col, "pickup_datetime"}
    numerical_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Run Analysis
    analyze_target(df, target_col)
    analyze_tabular_input(df, numerical_cols)
    analyze_relationships(df, target_col, numerical_cols)


if __name__ == "__main__":
    main()
