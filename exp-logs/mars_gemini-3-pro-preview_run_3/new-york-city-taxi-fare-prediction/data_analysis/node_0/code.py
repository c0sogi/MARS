import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# 1. Setup and Configuration
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Silence warnings
    warnings.filterwarnings("ignore")
    set_seed(42)

    # Define paths
    TRAIN_PATH = "./metadata/train.parquet"

    # 2. Load Data
    # We load the full dataset for accurate statistics, but will sample for heavy compute
    try:
        df = pd.read_parquet(TRAIN_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Preprocessing (Basic Type Conversion)
    # Convert pickup_datetime to actual datetime object for analysis
    if "pickup_datetime" in df.columns:
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

    # Define Column Types
    target_col = "fare_amount"

    # Identify numerical columns (excluding target and key)
    # We know the schema from the problem description
    numerical_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]

    # ==========================================
    # SECTION 1: TARGET VARIABLE ANALYSIS
    # ==========================================
    print("TARGET VARIABLE ANALYSIS")

    target_data = df[target_col].dropna()

    # Distribution stats
    t_skew = skew(target_data)
    t_kurt = kurtosis(target_data)

    print(f"Target Variable: {target_col}")
    print(f"Skewness: {t_skew:.4f}")
    print(f"Kurtosis: {t_kurt:.4f}")
    print("-" * 30)

    # ==========================================
    # SECTION 2: INPUT DATA ANALYSIS (TABULAR)
    # ==========================================
    print("INPUT DATA ANALYSIS")

    # A. Missing Values
    print("Missing Values:")
    missing_counts = df.isnull().sum()
    total_rows = len(df)
    for col, count in missing_counts.items():
        pct = (count / total_rows) * 100
        print(f"{col}: {count} ({pct:.4f}%)")
    print("-" * 20)

    # B. Numerical Analysis
    print("Numerical Features Statistics:")
    # We calculate stats on the full dataset
    stats = df[numerical_cols].describe().T[["mean", "std", "min", "max"]]

    # Outlier Analysis using IQR
    # Doing this on full dataset might be memory intensive if we create masks for all cols at once.
    # We iterate.
    outlier_counts = {}
    for col in numerical_cols:
        series = df[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        outlier_counts[col] = outliers

    for col in numerical_cols:
        row = stats.loc[col]
        n_outliers = outlier_counts[col]
        print(f"Feature: {col}")
        print(f"  Mean: {row['mean']:.4f}, Std: {row['std']:.4f}")
        print(f"  Min:  {row['min']:.4f}, Max: {row['max']:.4f}")
        print(f"  Outliers (IQR method): {n_outliers}")
    print("-" * 20)

    # C. Categorical / Temporal Analysis
    # The main non-numerical feature is pickup_datetime.
    # We treat it as a source of categorical/cyclic features.
    print("Temporal Analysis (pickup_datetime):")
    if "pickup_datetime" in df.columns:
        min_date = df["pickup_datetime"].min()
        max_date = df["pickup_datetime"].max()
        print(f"  Range: {min_date} to {max_date}")

        # Check for unique years to see coverage
        unique_years = df["pickup_datetime"].dt.year.unique()
        print(f"  Years covered: {sorted(unique_years)}")
    else:
        print("  pickup_datetime column not found.")
    print("-" * 30)

    # ==========================================
    # SECTION 3: FEATURE/SIGNAL RELATIONSHIPS
    # ==========================================
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Sampling for expensive computations
    # We use 100,000 samples to keep it fast
    SAMPLE_SIZE = 100000
    if len(df) > SAMPLE_SIZE:
        # Stratified sample based on fare_amount bins to preserve distribution
        # Simple binning for stratification
        try:
            df["fare_bin"] = pd.qcut(
                df[target_col], q=10, labels=False, duplicates="drop"
            )
            sample_df = df.groupby("fare_bin", group_keys=False).apply(
                lambda x: x.sample(min(len(x), int(SAMPLE_SIZE / 10)), random_state=42)
            )
            # If groupby apply results in fewer than expected (due to small bins), just random sample remainder
            if len(sample_df) < SAMPLE_SIZE:
                remaining = SAMPLE_SIZE - len(sample_df)
                others = df.drop(sample_df.index).sample(remaining, random_state=42)
                sample_df = pd.concat([sample_df, others])

            # Clean up temp column
            df.drop(columns=["fare_bin"], inplace=True)
            if "fare_bin" in sample_df.columns:
                sample_df.drop(columns=["fare_bin"], inplace=True)

        except Exception:
            # Fallback to simple random sample if stratification fails
            sample_df = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        sample_df = df.copy()

    # Feature Engineering on Sample for Analysis
    # 1. Temporal Features
    sample_df["hour"] = sample_df["pickup_datetime"].dt.hour
    sample_df["day_of_week"] = sample_df["pickup_datetime"].dt.dayofweek
    sample_df["year"] = sample_df["pickup_datetime"].dt.year

    # 2. Spatial Features (Haversine Distance)
    # Approximate radius of earth in km
    R = 6371.0

    lat1 = np.radians(sample_df["pickup_latitude"])
    lon1 = np.radians(sample_df["pickup_longitude"])
    lat2 = np.radians(sample_df["dropoff_latitude"])
    lon2 = np.radians(sample_df["dropoff_longitude"])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    sample_df["haversine_dist"] = R * c

    # Prepare data for Correlation and RF
    # Drop non-numeric for correlation (key, pickup_datetime)
    analysis_cols = numerical_cols + [
        "hour",
        "day_of_week",
        "year",
        "haversine_dist",
        target_col,
    ]
    # Filter out NaNs created by lag/diff or existing NaNs
    analysis_df = sample_df[analysis_cols].dropna()

    # A. Correlation
    print("Correlation Analysis (Pearson):")
    corr_matrix = analysis_df.corr(method="pearson")

    # Check for redundancy (Correlation > 0.90)
    # We look at upper triangle only
    redundant_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) > 0.90:
                redundant_pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

    if redundant_pairs:
        print("Redundant Features (Correlation > 0.90):")
        for f1, f2, val in redundant_pairs:
            print(f"  {f1} - {f2}: {val:.4f}")
    else:
        print("  No highly collinear pairs (> 0.90) found.")

    # Correlations with Target
    print(f"\nTop Correlations with Target ({target_col}):")
    target_corr = (
        corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    )
    for feat, val in target_corr.head(5).items():
        print(f"  {feat}: {val:.4f}")
    print("-" * 20)

    # B. Feature Importance (Random Forest)
    print("Feature Importance (Random Forest):")
    X = analysis_df.drop(columns=[target_col])
    y = analysis_df[target_col]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42, verbose=0
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 5 Important Features:")
    for i in range(min(5, len(X.columns))):
        feat_name = X.columns[indices[i]]
        imp_val = importances[indices[i]]
        print(f"  {i+1}. {feat_name}: {imp_val:.4f}")


if __name__ == "__main__":
    main()
