import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
SEED = 42
SAMPLE_SIZE = 200  # Number of files to sample for detailed sensor analysis

# Set seeds for reproducibility
np.random.seed(SEED)


def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def analyze_target(df):
    print_section("Target Variable Analysis")
    target = df["time_to_eruption"]

    print("Variable: time_to_eruption")
    print(f"Type: Regression (Continuous)")
    print(f"Count: {len(target)}")
    print(f"Mean: {target.mean():.4f}")
    print(f"Std Dev: {target.std():.4f}")
    print(f"Min: {target.min():.4f}")
    print(f"Max: {target.max():.4f}")

    # Normality Check
    target_skew = skew(target)
    target_kurt = kurtosis(target)
    print(f"Skewness: {target_skew:.4f} (Values > 1 indicate high skew)")
    print(f"Kurtosis: {target_kurt:.4f}")


def load_sample_data(meta_df):
    """
    Loads a random sample of sensor files into a single DataFrame.
    Returns the concatenated raw dataframe and a dataframe aggregated by segment_id.
    """
    sample_meta = meta_df.sample(n=min(SAMPLE_SIZE, len(meta_df)), random_state=SEED)

    raw_data_list = []
    agg_data_list = []

    file_lengths = []

    for _, row in sample_meta.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        try:
            # Load sensor data
            # Using float32 to handle potential NaNs and save memory
            df = pd.read_csv(file_path, dtype="float32")

            # Check length
            file_lengths.append(len(df))

            # Add segment_id for aggregation later
            df["segment_id"] = row["segment_id"]

            # Append to raw list (for global stats)
            raw_data_list.append(df)

            # Compute aggregates for Feature Relationship analysis
            # We calculate mean and std for each sensor for this segment
            stats = df.drop(columns=["segment_id"]).agg(["mean", "std"])
            # Flatten the stats
            flat_stats = stats.unstack().to_frame().T
            flat_stats.columns = [f"{col[0]}_{col[1]}" for col in flat_stats.columns]
            flat_stats["segment_id"] = row["segment_id"]
            flat_stats["time_to_eruption"] = row["time_to_eruption"]
            agg_data_list.append(flat_stats)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    raw_df = pd.concat(raw_data_list, ignore_index=True)
    agg_df = pd.concat(agg_data_list, ignore_index=True)

    return raw_df, agg_df, file_lengths


def analyze_input_data(raw_df, file_lengths):
    print_section("Input Data Analysis (Tabular/Signal)")

    # 1. Signal Consistency
    print("--- Signal Consistency ---")
    unique_lengths = np.unique(file_lengths)
    print(f"Unique File Lengths (Rows): {unique_lengths}")
    if len(unique_lengths) == 1:
        print("Consistency: All sampled files have the same duration.")
    else:
        print("Consistency: Variable file lengths detected.")

    # 2. Numerical Stats per Sensor
    print("\n--- Numerical Statistics (Global Sample) ---")
    sensor_cols = [c for c in raw_df.columns if c.startswith("sensor")]

    stats_df = raw_df[sensor_cols].describe().T[["mean", "std", "min", "max"]]

    # Calculate Outliers (IQR Method)
    # Note: Doing this on the full concatenated sample
    Q1 = raw_df[sensor_cols].quantile(0.25)
    Q3 = raw_df[sensor_cols].quantile(0.75)
    IQR = Q3 - Q1

    outlier_counts = (
        (raw_df[sensor_cols] < (Q1 - 1.5 * IQR))
        | (raw_df[sensor_cols] > (Q3 + 1.5 * IQR))
    ).sum()
    stats_df["outlier_count"] = outlier_counts
    stats_df["outlier_pct"] = (outlier_counts / len(raw_df)) * 100

    # Format and print
    print(
        f"{'Sensor':<12} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Outliers':<10} {'Outlier %':<10}"
    )
    for idx, row in stats_df.iterrows():
        print(
            f"{idx:<12} {row['mean']:<12.4f} {row['std']:<12.4f} {row['min']:<12.4f} {row['max']:<12.4f} {int(row['outlier_count']):<10} {row['outlier_pct']:.2f}%"
        )

    # 3. Missing Values
    print("\n--- Missing Values ---")
    missing = raw_df[sensor_cols].isnull().sum()
    missing_pct = (missing / len(raw_df)) * 100

    print(f"{'Sensor':<12} {'Missing Count':<15} {'Missing %':<10}")
    has_missing = False
    for idx, count in missing.items():
        pct = missing_pct[idx]
        if count > 0:
            has_missing = True
        print(f"{idx:<12} {count:<15} {pct:.4f}%")

    if not has_missing:
        print("No missing values detected in the sample.")


def analyze_relationships(agg_df):
    print_section("Feature/Signal Relationships")

    # Prepare data
    feature_cols = [
        c for c in agg_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X = agg_df[feature_cols]
    y = agg_df["time_to_eruption"]

    # 1. Correlation with Target
    print("--- Correlation with Target (Top 5) ---")
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    for feat, corr in correlations.head(5).items():
        print(f"{feat}: {corr:.4f}")

    # 2. Feature Importance (Random Forest)
    print("\n--- Random Forest Feature Importance (Top 5) ---")
    rf = RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1)
    rf.fit(
        X.fillna(0), y
    )  # Handle NaNs for RF by filling 0 (simple imputation for importance check)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    for feat, imp in importances.head(5).items():
        print(f"{feat}: {imp:.4f}")

    # 3. Redundancy (Collinear Pairs)
    print("\n--- Feature Redundancy (Correlation > 0.90) ---")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr_pairs = []
    for column in upper.columns:
        for idx, val in upper[column].items():
            if val > 0.90:
                high_corr_pairs.append((idx, column, val))

    if high_corr_pairs:
        # Sort by correlation strength
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        for f1, f2, val in high_corr_pairs[:10]:  # Print top 10
            print(f"{f1} - {f2}: {val:.4f}")
        if len(high_corr_pairs) > 10:
            print(f"... and {len(high_corr_pairs) - 10} more pairs.")
    else:
        print("No highly collinear pairs found.")

    # 4. Metadata Relationship
    print("\n--- Metadata Relationship ---")
    # Check if 'mean' of signals correlates with target
    # We already did this in section 1, but let's summarize the direction
    # e.g. "Do higher sensor readings correlate with longer time to eruption?"

    # Take the most important feature
    top_feat = importances.index[0]
    direction = "Positive" if X[top_feat].corr(y) > 0 else "Negative"
    print(
        f"Primary Insight: The feature '{top_feat}' has a {direction} correlation with time_to_eruption."
    )


def main():
    # 1. Load Metadata
    if not os.path.exists(TRAIN_META_PATH):
        print(f"Error: Metadata file not found at {TRAIN_META_PATH}")
        return

    train_df = pd.read_csv(TRAIN_META_PATH)

    # 2. Target Analysis
    analyze_target(train_df)

    # 3. Load Sample Data
    print(f"\nSampling {SAMPLE_SIZE} files for detailed input analysis...")
    raw_df, agg_df, file_lengths = load_sample_data(train_df)

    # 4. Input Data Analysis
    analyze_input_data(raw_df, file_lengths)

    # 5. Relationship Analysis
    analyze_relationships(agg_df)


if __name__ == "__main__":
    main()
