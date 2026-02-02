import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
METADATA_FILE = "./metadata/train.csv"
INPUT_DIR = "./input"
SEED = 42
SAMPLE_SIZE = 400  # Number of files to sample for detailed signal analysis

# Set fixed random seeds
np.random.seed(SEED)


def run_eda():
    # 1. Load Metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file {METADATA_FILE} not found.")
        return

    df_train = pd.read_csv(METADATA_FILE)

    # ---------------------------------------------------------
    # TARGET VARIABLE ANALYSIS
    # ---------------------------------------------------------
    print("TARGET VARIABLE ANALYSIS")
    target = df_train["time_to_eruption"]

    print(f"Target Variable: time_to_eruption")
    print(f"Type: Regression")
    print(f"Count: {len(target)}")
    print(f"Mean: {target.mean():.4f}")
    print(f"Std Dev: {target.std():.4f}")
    print(f"Min: {target.min():.4f}")
    print(f"Max: {target.max():.4f}")

    # Skewness and Kurtosis
    target_skew = skew(target)
    target_kurt = kurtosis(target)
    print(f"Skewness: {target_skew:.4f}")
    print(f"Kurtosis: {target_kurt:.4f}")
    print("-" * 30)

    # ---------------------------------------------------------
    # INPUT DATA ANALYSIS (TABULAR/SIGNAL)
    # ---------------------------------------------------------
    print("INPUT DATA ANALYSIS (SENSOR READINGS)")

    # Sample files for analysis
    if len(df_train) > SAMPLE_SIZE:
        df_sample = df_train.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    else:
        df_sample = df_train.copy()

    print(f"Analysis performed on a random sample of {len(df_sample)} segments.")

    # Accumulators
    feature_rows = []
    target_rows = []
    nan_counts_per_segment = []

    # To compute global stats roughly
    global_stats_accum = {
        f"sensor_{i}": {
            "means": [],
            "stds": [],
            "mins": [],
            "maxs": [],
            "nans": 0,
            "count": 0,
        }
        for i in range(1, 11)
    }

    for idx, row in df_sample.iterrows():
        file_path = os.path.join(INPUT_DIR, row["file_path"])
        try:
            df_sensor = pd.read_csv(file_path)

            # Meta-feature: NaNs in this file
            n_nans = df_sensor.isna().sum().sum()
            nan_counts_per_segment.append(n_nans)

            # Feature Extraction (per segment)
            segment_feats = {}

            for i in range(1, 11):
                col = f"sensor_{i}"
                if col in df_sensor.columns:
                    series = df_sensor[col]

                    # Stats ignoring NaNs
                    s_mean = series.mean()
                    s_std = series.std()
                    s_min = series.min()
                    s_max = series.max()
                    s_nans = series.isna().sum()

                    # Update Global Accumulators
                    if not np.isnan(s_mean):
                        global_stats_accum[col]["means"].append(s_mean)
                    if not np.isnan(s_std):
                        global_stats_accum[col]["stds"].append(s_std)
                    if not np.isnan(s_min):
                        global_stats_accum[col]["mins"].append(s_min)
                    if not np.isnan(s_max):
                        global_stats_accum[col]["maxs"].append(s_max)
                    global_stats_accum[col]["nans"] += s_nans
                    global_stats_accum[col]["count"] += len(series)

                    # Store Features for Relationship Analysis
                    segment_feats[f"{col}_mean"] = 0 if np.isnan(s_mean) else s_mean
                    segment_feats[f"{col}_std"] = 0 if np.isnan(s_std) else s_std
                    segment_feats[f"{col}_min"] = 0 if np.isnan(s_min) else s_min
                    segment_feats[f"{col}_max"] = 0 if np.isnan(s_max) else s_max

            feature_rows.append(segment_feats)
            target_rows.append(row["time_to_eruption"])

        except Exception as e:
            continue

    # Report Global Stats
    print(
        f"{'Sensor':<12} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'NaN %':<12}"
    )

    for i in range(1, 11):
        col = f"sensor_{i}"
        stats = global_stats_accum[col]

        # Aggregation logic
        # Global Mean ~ Mean of means (exact if all files same length, which they are: 60001)
        g_mean = np.mean(stats["means"]) if stats["means"] else 0
        # Global Std ~ Average of stds (approximation of signal volatility)
        g_std = np.mean(stats["stds"]) if stats["stds"] else 0
        g_min = np.min(stats["mins"]) if stats["mins"] else 0
        g_max = np.max(stats["maxs"]) if stats["maxs"] else 0

        nan_pct = (stats["nans"] / stats["count"] * 100) if stats["count"] > 0 else 0

        print(
            f"{col:<12} {g_mean:<12.4f} {g_std:<12.4f} {g_min:<12.4f} {g_max:<12.4f} {nan_pct:<12.4f}"
        )

    print("-" * 30)

    # ---------------------------------------------------------
    # FEATURE/SIGNAL RELATIONSHIPS
    # ---------------------------------------------------------
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Create DataFrame for analysis
    X = pd.DataFrame(feature_rows)
    y = pd.Series(target_rows, name="target")

    # Fill any remaining NaNs in features (e.g. if a sensor was totally dead)
    X = X.fillna(0)

    # A. Structured Relationships
    print("--- Structured Relationships ---")

    # 1. Correlation
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print("Top 5 Features Correlated with Target (Pearson):")
    for feat, corr in correlations.head(5).items():
        print(f"{feat}: {corr:.4f}")

    # 2. Importance (Random Forest)
    print("\nTop 5 Important Features (Random Forest):")
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    for feat, imp in importances.head(5).items():
        print(f"{feat}: {imp:.4f}")

    # 3. Redundancy
    print("\nRedundant Feature Pairs (Correlation > 0.90):")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find pairs
    redundant_pairs = []
    for col in upper.columns:
        high_corr_rows = upper.index[upper[col] > 0.90].tolist()
        for row in high_corr_rows:
            redundant_pairs.append((row, col, upper.loc[row, col]))

    # Sort by correlation strength
    redundant_pairs.sort(key=lambda x: x[2], reverse=True)

    if redundant_pairs:
        for p in redundant_pairs[:5]:
            print(f"{p[0]} - {p[1]}: {p[2]:.4f}")
        if len(redundant_pairs) > 5:
            print(f"... and {len(redundant_pairs) - 5} more pairs.")
    else:
        print("No redundant pairs found.")

    # B. Unstructured (Meta-Feature) Relationships
    print("\n--- Meta-Feature Relationships ---")

    # Analyze relationship between Missing Values and Target
    df_meta_analysis = pd.DataFrame(
        {"nans": nan_counts_per_segment, "target": target_rows}
    )

    # Correlation
    nan_corr = df_meta_analysis["nans"].corr(df_meta_analysis["target"])
    print(f"Correlation between Segment NaN Count and Target: {nan_corr:.4f}")

    # Group comparison
    has_nans = df_meta_analysis[df_meta_analysis["nans"] > 0]
    no_nans = df_meta_analysis[df_meta_analysis["nans"] == 0]

    mean_target_nans = has_nans["target"].mean() if not has_nans.empty else 0
    mean_target_clean = no_nans["target"].mean() if not no_nans.empty else 0

    print(
        f"Mean Target (Segments with NaNs): {mean_target_nans:.4f} (Count: {len(has_nans)})"
    )
    print(
        f"Mean Target (Segments w/o NaNs):  {mean_target_clean:.4f} (Count: {len(no_nans)})"
    )


if __name__ == "__main__":
    run_eda()
