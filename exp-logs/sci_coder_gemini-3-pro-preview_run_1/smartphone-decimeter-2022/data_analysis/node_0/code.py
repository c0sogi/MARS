import os
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
METADATA_PATH = "./metadata/train_metadata.csv"
INPUT_DIR = "./input"
SEED = 42
SAMPLE_DRIVES_COUNT = 3  # Number of drives to sample for detailed input analysis


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_header(title):
    print(f"\n{'=' * 80}")
    print(f"{title.upper()}")
    print(f"{'=' * 80}")


def analyze_targets(df_meta):
    print_header("2. Target Variable Analysis")

    targets = ["LatitudeDegrees", "LongitudeDegrees"]

    for target in targets:
        data = df_meta[target].dropna()

        # Distribution stats
        mean_val = data.mean()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()

        # Normality checks
        skew = data.skew()
        kurt = data.kurtosis()

        print(f"\nTarget: {target}")
        print(f"  Count: {len(data)}")
        print(f"  Mean: {mean_val:.4f} | Std: {std_val:.4f}")
        print(f"  Min:  {min_val:.4f} | Max: {max_val:.4f}")
        print(f"  Skewness: {skew:.4f} (Normal=0)")
        print(f"  Kurtosis: {kurt:.4f} (Normal=3 for Pearson, 0 for Fisher)")

        if abs(skew) > 1:
            print("  -> Distribution is highly skewed.")
        else:
            print("  -> Distribution is approximately symmetric.")


def analyze_tabular_column(df, col_name, col_type="numerical"):
    if col_type == "numerical":
        data = df[col_name].dropna()
        if len(data) == 0:
            print(f"  {col_name}: All NaN")
            return

        mean_val = data.mean()
        std_val = data.std()

        # Outliers (IQR)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()

        print(f"  {col_name}:")
        print(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"    Min: {data.min():.4f}, Max: {data.max():.4f}")
        print(f"    Outliers (IQR): {outliers} ({outliers/len(data)*100:.2f}%)")

    elif col_type == "categorical":
        data = df[col_name].astype(str)
        unique_count = data.nunique()
        print(f"  {col_name}:")
        print(f"    Cardinality: {unique_count}")

        if unique_count > 50:
            print("    -> High cardinality column (>50 categories).")

        # Check for rare labels
        value_counts = data.value_counts(normalize=True)
        rare_labels = value_counts[value_counts < 0.01]
        if not rare_labels.empty:
            print(f"    -> Rare labels (<1%): {len(rare_labels)} categories")


def analyze_missing(df):
    print("\n  Missing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    total = len(df)
    if missing.empty:
        print("    None")
    else:
        for col, count in missing.items():
            print(f"    {col}: {count} ({count/total*100:.2f}%)")


def analyze_inputs(df_meta):
    print_header("3. Input Data Analysis (Tabular)")

    # Sample specific drives to analyze raw data
    all_drives = df_meta["drive_id"].unique()
    sample_drives = np.random.choice(
        all_drives, min(len(all_drives), SAMPLE_DRIVES_COUNT), replace=False
    )

    print(
        f"Sampling {len(sample_drives)} drives for detailed sensor analysis: {sample_drives}"
    )

    gnss_dfs = []
    imu_dfs = []

    for drive_id in sample_drives:
        # Get one phone from this drive to avoid duplicating shared drive data if multiple phones exist
        drive_subset = df_meta[df_meta["drive_id"] == drive_id]
        if drive_subset.empty:
            continue

        # Just pick the first phone's data for this drive
        row = drive_subset.iloc[0]

        gnss_path = os.path.join(INPUT_DIR, row["gnss_path"])
        imu_path = os.path.join(INPUT_DIR, row["imu_path"])

        if os.path.exists(gnss_path):
            try:
                # Read a subset of columns to save memory if needed, but these files aren't huge
                g_df = pd.read_csv(gnss_path)
                gnss_dfs.append(g_df)
            except Exception as e:
                print(f"Error reading {gnss_path}: {e}")

        if os.path.exists(imu_path):
            try:
                i_df = pd.read_csv(imu_path)
                imu_dfs.append(i_df)
            except Exception as e:
                print(f"Error reading {imu_path}: {e}")

    # --- GNSS Analysis ---
    if gnss_dfs:
        full_gnss = pd.concat(gnss_dfs, ignore_index=True)
        print("\n[GNSS Data Analysis]")
        print(f"  Total Rows (Sampled): {len(full_gnss)}")

        # Numerical Columns of Interest
        num_cols = [
            "Cn0DbHz",
            "SvElevationDegrees",
            "SvAzimuthDegrees",
            "RawPseudorangeMeters",
        ]
        for col in num_cols:
            if col in full_gnss.columns:
                analyze_tabular_column(full_gnss, col, "numerical")

        # Categorical Columns
        cat_cols = ["ConstellationType", "MultipathIndicator", "CodeType"]
        for col in cat_cols:
            if col in full_gnss.columns:
                analyze_tabular_column(full_gnss, col, "categorical")

        analyze_missing(full_gnss[num_cols + cat_cols])

    # --- IMU Analysis ---
    if imu_dfs:
        full_imu = pd.concat(imu_dfs, ignore_index=True)
        print("\n[IMU Data Analysis]")
        print(f"  Total Rows (Sampled): {len(full_imu)}")

        # IMU data is often mixed in one file with a MessageType column
        if "MessageType" in full_imu.columns:
            msg_types = full_imu["MessageType"].unique()
            print(f"  Message Types found: {msg_types}")

            for m_type in msg_types:
                subset = full_imu[full_imu["MessageType"] == m_type]
                print(f"\n  Analysis for {m_type} ({len(subset)} rows):")
                # Usually MeasurementX, Y, Z
                cols = [c for c in subset.columns if "Measurement" in c]
                for col in cols:
                    analyze_tabular_column(subset, col, "numerical")
        else:
            print("  'MessageType' column not found in IMU data.")


def analyze_relationships(df_meta):
    print_header("4. Feature/Signal Relationships")

    # We need to link input features to targets.
    # Since raw data is high frequency and targets are 1Hz, we aggregate raw data.

    all_drives = df_meta["drive_id"].unique()
    sample_drives = np.random.choice(
        all_drives, min(len(all_drives), 2), replace=False
    )  # Use 2 drives for relationship

    combined_data = []

    print(
        f"Aggregating data from {len(sample_drives)} drives for correlation analysis..."
    )

    for drive_id in sample_drives:
        drive_subset = df_meta[df_meta["drive_id"] == drive_id]

        for _, row in drive_subset.iterrows():
            gnss_path = os.path.join(INPUT_DIR, row["gnss_path"])

            if not os.path.exists(gnss_path):
                continue

            # Load GNSS
            gnss_df = pd.read_csv(gnss_path)

            # Simple Aggregation: Group by utcTimeMillis (approx match to UnixTimeMillis)
            # Note: utcTimeMillis in GNSS might need matching logic, but usually they are close.
            # We will group by unique epochs in GNSS.

            # Features to extract per epoch
            gnss_agg = gnss_df.groupby("utcTimeMillis").agg(
                {
                    "Cn0DbHz": [
                        "mean",
                        "max",
                        "count",
                    ],  # Signal strength and num satellites
                    "SvElevationDegrees": "mean",
                    "RawPseudorangeMeters": "std",  # Variability in range
                }
            )

            gnss_agg.columns = [
                "_".join(col).strip() for col in gnss_agg.columns.values
            ]
            gnss_agg = gnss_agg.reset_index()
            gnss_agg.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

            # Merge with Target (Ground Truth)
            # GT is in df_meta, but we need the specific rows for this phone/drive
            # We can just use the row's metadata info, but we need the full GT sequence.
            # The df_meta passed in IS the training metadata which contains GT targets.

            # Filter meta for this specific phone run
            gt_subset = df_meta[
                (df_meta["drive_id"] == row["drive_id"])
                & (df_meta["phone_name"] == row["phone_name"])
            ].copy()

            # Merge
            # Using inner join to find matching timestamps
            merged = pd.merge(gt_subset, gnss_agg, on="UnixTimeMillis", how="inner")

            if not merged.empty:
                combined_data.append(merged)

    if not combined_data:
        print("Could not merge any data for relationship analysis.")
        return

    full_df = pd.concat(combined_data, ignore_index=True)
    print(f"Constructed dataset with {len(full_df)} samples.")

    # Define features and target
    feature_cols = [c for c in full_df.columns if "Cn0" in c or "Sv" in c or "Raw" in c]
    target_col = "LatitudeDegrees"  # Analyze relationship with one target for brevity

    if not feature_cols:
        print("No features extracted.")
        return

    # 1. Correlation
    print("\n[Correlation Analysis (Top 5 vs Latitude)]")
    corrs = full_df[feature_cols + [target_col]].corr(method="pearson")
    target_corr = corrs[target_col].drop(target_col).abs().sort_values(ascending=False)
    print(target_corr.head(5))

    # 2. Collinearity
    print("\n[Collinearity Check (Corr > 0.90)]")
    feature_corrs = full_df[feature_cols].corr().abs()
    # Select upper triangle
    upper = feature_corrs.where(np.triu(np.ones(feature_corrs.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    if to_drop:
        print(f"  Highly correlated features found: {to_drop}")
    else:
        print("  No highly correlated feature pairs found.")

    # 3. Feature Importance (Random Forest)
    print("\n[Feature Importance (Random Forest)]")
    X = full_df[feature_cols].fillna(0)
    y = full_df[target_col]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    print("  Top 5 Features:")
    for feat, imp in importances.head(5).items():
        print(f"    {feat}: {imp:.4f}")


def main():
    set_seed(SEED)

    # 1. Data Integrity Check (Implicit by using provided metadata)
    if not os.path.exists(METADATA_PATH):
        print(
            "Metadata file not found. Please ensure generation script ran successfully."
        )
        return

    df_meta = pd.read_csv(METADATA_PATH)

    # 2. Target Analysis
    analyze_targets(df_meta)

    # 3. Input Data Analysis
    analyze_inputs(df_meta)

    # 4. Relationship Analysis
    analyze_relationships(df_meta)

    print_header("Analysis Complete")


if __name__ == "__main__":
    main()
