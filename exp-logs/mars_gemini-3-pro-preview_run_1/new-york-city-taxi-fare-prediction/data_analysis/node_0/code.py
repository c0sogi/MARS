import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
import os

# --- Configuration ---
METADATA_PATH = "./metadata/train.parquet"
SAMPLE_SIZE_RELATIONSHIPS = 100000  # Sample size for expensive ops (RF, Correlation)
RANDOM_SEED = 42

# --- Setup ---
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


def print_header(title):
    print(f"\n{'='*10} {title} {'='*10}")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def analyze_target(df, target_col):
    print_header("TARGET VARIABLE ANALYSIS")
    target = df[target_col]

    # Basic Stats
    print(f"Target Variable: '{target_col}'")
    print(f"Count: {len(target)}")
    print(f"Mean: {target.mean():.4f}")
    print(f"Std Dev: {target.std():.4f}")
    print(f"Min: {target.min():.4f}")
    print(f"Max: {target.max():.4f}")

    # Normality
    target_skew = skew(target)
    target_kurt = kurtosis(target)
    print(f"Skewness: {target_skew:.4f} (Positively skewed expected for prices)")
    print(f"Kurtosis: {target_kurt:.4f}")

    # Domain specific checks
    neg_fares = (target < 0).sum()
    zero_fares = (target == 0).sum()
    print(f"Negative values count: {neg_fares} ({neg_fares/len(target)*100:.4f}%)")
    print(f"Zero values count: {zero_fares} ({zero_fares/len(target)*100:.4f}%)")


def analyze_tabular_inputs(df, target_col):
    print_header("INPUT DATA ANALYSIS (TABULAR)")

    # Identify feature types
    # exclude key, target, and datetime for raw numerical analysis
    exclude_cols = ["key", target_col, "pickup_datetime"]
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    print("--- Numerical Features ---")
    for col in num_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()

        print(f"Feature: {col}")
        print(f"  Mean: {series.mean():.4f} | Std: {series.std():.4f}")
        print(f"  Min: {series.min():.4f} | Max: {series.max():.4f}")
        print(f"  Outliers (IQR method): {outliers} ({outliers/len(series)*100:.2f}%)")

        # Domain specific check for coordinates
        if "latitude" in col:
            invalid_lat = ((series < -90) | (series > 90)).sum()
            if invalid_lat > 0:
                print(
                    f"  [WARNING] Invalid Latitude values (<-90 or >90): {invalid_lat}"
                )
        if "longitude" in col:
            invalid_lon = ((series < -180) | (series > 180)).sum()
            if invalid_lon > 0:
                print(
                    f"  [WARNING] Invalid Longitude values (<-180 or >180): {invalid_lon}"
                )
        print("-" * 30)

    print("\n--- Categorical / Text Features ---")
    # In this dataset, 'key' is ID and 'pickup_datetime' is time.
    # We check cardinality just in case.
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        if col == "key":
            continue  # Skip ID
        if col == "pickup_datetime":
            print(f"Feature: {col} (Timestamp)")
            continue

        unique_count = df[col].nunique()
        print(f"Feature: {col}")
        print(f"  Cardinality: {unique_count}")
        if unique_count > 50:
            print(f"  [INFO] High cardinality (>50 categories).")

    print("\n--- Missing Values ---")
    nan_counts = df.isna().sum()
    total_rows = len(df)
    for col, count in nan_counts.items():
        if count > 0:
            print(f"{col}: {count} NaNs ({count/total_rows*100:.4f}%)")
    if nan_counts.sum() == 0:
        print("No missing values found in the dataset.")


def analyze_relationships(df, target_col):
    print_header("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Preprocessing for Relationship Analysis
    # We need to process the datetime and create a distance feature to get meaningful relationships.
    # Working on a sample to save time/memory
    if len(df) > SAMPLE_SIZE_RELATIONSHIPS:
        print(f"Sampling {SAMPLE_SIZE_RELATIONSHIPS} rows for relationship analysis...")
        df_sample = df.sample(
            n=SAMPLE_SIZE_RELATIONSHIPS, random_state=RANDOM_SEED
        ).copy()
    else:
        df_sample = df.copy()

    # Convert datetime
    df_sample["pickup_datetime"] = pd.to_datetime(
        df_sample["pickup_datetime"], utc=True
    )

    # Feature Engineering (Temp for EDA)
    df_sample["hour"] = df_sample["pickup_datetime"].dt.hour
    df_sample["year"] = df_sample["pickup_datetime"].dt.year
    df_sample["month"] = df_sample["pickup_datetime"].dt.month
    df_sample["weekday"] = df_sample["pickup_datetime"].dt.dayofweek

    # Calculate Distance (Crucial for Taxi data)
    df_sample["distance_km"] = haversine_distance(
        df_sample["pickup_latitude"],
        df_sample["pickup_longitude"],
        df_sample["dropoff_latitude"],
        df_sample["dropoff_longitude"],
    )

    # Drop non-numeric for correlation/RF
    drop_cols = ["key", "pickup_datetime"]
    analysis_df = df_sample.drop(
        columns=[c for c in drop_cols if c in df_sample.columns]
    )

    # Handle infinite distances if any (division by zero or bad coords)
    analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan).dropna()

    # 2. Structured Relationships
    print("\n--- Correlation (Pearson) ---")
    corr_matrix = analysis_df.corr(method="pearson")
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    print(f"Top correlations with {target_col}:")
    print(target_corr.drop(target_col).head(5).to_string(float_format="%.4f"))
    print("\nBottom correlations with {target_col}:")
    print(target_corr.drop(target_col).tail(5).to_string(float_format="%.4f"))

    print("\n--- Redundancy (Collinearity > 0.90) ---")
    # Get pairs
    high_corr_pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) > 0.90:
                high_corr_pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

    if high_corr_pairs:
        for c1, c2, val in high_corr_pairs:
            print(f"{c1} & {c2}: {val:.4f}")
    else:
        print("No highly collinear pairs found.")

    # 3. Feature Importance (Random Forest)
    print("\n--- Feature Importance (Random Forest) ---")
    X = analysis_df.drop(columns=[target_col])
    y = analysis_df[target_col]

    rf = RandomForestRegressor(
        n_estimators=10, max_depth=10, n_jobs=-1, random_state=RANDOM_SEED
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("Top 5 Important Features:")
    print(importances.head(5).to_string(float_format="%.4f"))

    # 4. Unstructured/Meta Relationships
    # Here we check if the derived 'distance' (which combines 4 raw features)
    # explains the target better than raw features.
    print("\n--- Meta-Feature Analysis ---")
    dist_corr = analysis_df["distance_km"].corr(analysis_df[target_col])
    print(f"Correlation of Derived 'distance_km' with Target: {dist_corr:.4f}")
    if abs(dist_corr) > abs(target_corr.drop([target_col, "distance_km"]).max()):
        print(
            "Insight: Derived distance feature has higher correlation than any single raw coordinate."
        )


def main():
    print("Starting Exploratory Data Analysis...")

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: {METADATA_PATH} not found.")
        return

    # Using pyarrow for speed if available, else standard
    try:
        df = pd.read_parquet(METADATA_PATH)
    except Exception as e:
        print(f"Failed to load parquet: {e}")
        return

    print(f"Loaded dataset with shape: {df.shape}")

    target_col = "fare_amount"

    # Run Analysis
    analyze_target(df, target_col)
    analyze_tabular_inputs(df, target_col)
    analyze_relationships(df, target_col)

    print_header("EDA COMPLETE")


if __name__ == "__main__":
    main()
