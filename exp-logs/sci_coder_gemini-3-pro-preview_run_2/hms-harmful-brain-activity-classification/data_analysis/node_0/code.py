import pandas as pd
import numpy as np
import os
import random
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SAMPLE_SIZE = (
    100  # Number of files to sample for heavy I/O operations (EEG/Spectrograms)
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(RANDOM_STATE)
warnings.filterwarnings("ignore")


def print_section(title):
    print(f"\n{'='*20} {title.upper()} {'='*20}")


def load_parquet_file(relative_path):
    """Helper to load parquet files handling the input directory structure."""
    full_path = os.path.join(INPUT_DIR, relative_path)
    try:
        return pd.read_parquet(full_path)
    except Exception as e:
        return None


# -----------------------------------------------------------------------------
# Main Analysis Logic
# -----------------------------------------------------------------------------


def analyze_targets(df):
    print_section("Target Variable Analysis")

    # The targets are probabilities summing to 1.
    target_cols = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]

    # 1. Distribution of Probabilities
    print("Distribution of Target Probabilities (Mean across dataset):")
    means = df[target_cols].mean()
    for col, val in means.items():
        print(f"  {col:<15}: {val:.4f}")

    # 2. Class Balance (based on Max Probability / Hard Label)
    # We use the column with the highest probability as the 'class' for this stat
    df["dominant_class"] = df[target_cols].idxmax(axis=1)
    class_counts = df["dominant_class"].value_counts(normalize=True)

    print("\nClass Balance (Dominant Class Frequency):")
    for cls, freq in class_counts.items():
        print(f"  {cls:<15}: {freq:.4f}")

    # 3. Skewness/Kurtosis of the probabilities (Treating as regression targets)
    print("\nTarget Distribution Statistics (Skewness & Kurtosis):")
    for col in target_cols:
        s = skew(df[col])
        k = kurtosis(df[col])
        print(f"  {col:<15}: Skew={s:.4f}, Kurtosis={k:.4f}")


def analyze_tabular_metadata(df):
    print_section("Tabular Metadata Analysis")

    # Numerical Columns to analyze
    num_cols = [
        "total_votes",
        "eeg_label_offset_seconds",
        "spectogram_label_offset_seconds",
    ]
    # Filter only existing columns
    num_cols = [c for c in num_cols if c in df.columns]

    print("Numerical Feature Statistics:")
    if num_cols:
        stats = df[num_cols].describe().T[["mean", "std", "min", "max"]]
        # Calculate Outliers (IQR)
        outliers = {}
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            cnt = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outliers[col] = cnt

        for idx, row in stats.iterrows():
            print(
                f"  {idx:<30}: Mean={row['mean']:.4f}, Std={row['std']:.4f}, "
                f"Min={row['min']:.4f}, Max={row['max']:.4f}, Outliers={outliers[idx]}"
            )
    else:
        print("  No numerical metadata columns found.")

    # Categorical Columns
    cat_cols = ["expert_consensus", "patient_id"]
    print("\nCategorical Feature Statistics:")
    for col in cat_cols:
        if col in df.columns:
            nunique = df[col].nunique()
            print(f"  {col:<20}: {nunique} unique values.")
            if nunique > 50:
                print(f"    (High cardinality column)")

    # Missing Values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  No missing values found in metadata.")
    else:
        for col, val in missing.items():
            pct = (val / len(df)) * 100
            print(f"  {col:<20}: {val} ({pct:.2f}%)")


def analyze_eeg_data(df):
    print_section("EEG Data Analysis (Time-Series)")

    # Sample a subset of EEG files to analyze
    # We use eeg_path provided in metadata
    unique_eeg_paths = (
        df["eeg_path"]
        .drop_duplicates()
        .sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    )

    print(f"Analyzing a random sample of {len(unique_eeg_paths)} EEG files...")

    dims_list = []
    nan_ratios = []
    means = []
    stds = []
    durations = []

    sampling_rate = 200.0  # From dataset description

    for path in unique_eeg_paths:
        eeg_df = load_parquet_file(path)
        if eeg_df is None:
            continue

        # Dimensions
        dims_list.append(eeg_df.shape)

        # Duration
        durations.append(eeg_df.shape[0] / sampling_rate)

        # NaNs
        # EEG data often has NaNs for disconnected electrodes
        nan_count = eeg_df.isna().sum().sum()
        total_cells = eeg_df.size
        nan_ratios.append(nan_count / total_cells)

        # Pixel/Signal Stats (ignoring NaNs)
        vals = eeg_df.values.flatten()
        # Remove NaNs for stats calculation
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            means.append(np.mean(vals))
            stds.append(np.std(vals))

    # Report
    if dims_list:
        avg_rows = np.mean([d[0] for d in dims_list])
        avg_cols = np.mean([d[1] for d in dims_list])
        print(
            f"Average Dimensions: {avg_rows:.1f} time steps x {avg_cols:.1f} channels"
        )
        print(f"Average Duration  : {np.mean(durations):.4f} seconds")
        print(f"Global Signal Mean: {np.mean(means):.4f}")
        print(f"Global Signal Std : {np.mean(stds):.4f}")
        print(f"Average NaN Ratio : {np.mean(nan_ratios)*100:.4f}%")

        # Channel consistency check
        unique_channels = set([d[1] for d in dims_list])
        if len(unique_channels) == 1:
            print(
                f"Channel Consistency: All sampled files have {list(unique_channels)[0]} channels."
            )
        else:
            print(
                f"Channel Consistency: Inconsistent channel counts found: {unique_channels}"
            )

    # Return some meta-features for relationship analysis
    # We map path -> stats
    return pd.DataFrame(
        {
            "eeg_path": unique_eeg_paths.values,
            "eeg_mean": means,
            "eeg_std": stds,
            "eeg_nan_ratio": nan_ratios,
        }
    )


def analyze_spectrogram_data(df):
    print_section("Spectrogram Data Analysis (Image-like)")

    unique_spec_paths = (
        df["spectrogram_path"]
        .drop_duplicates()
        .sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    )

    print(f"Analyzing a random sample of {len(unique_spec_paths)} Spectrogram files...")

    widths = []
    heights = []
    means = []
    stds = []
    nan_ratios = []

    for path in unique_spec_paths:
        spec_df = load_parquet_file(path)
        if spec_df is None:
            continue

        # Parquet spectrograms: Columns are frequencies, Rows are time
        # Height = Time (rows), Width = Frequencies (columns)
        h, w = spec_df.shape
        widths.append(w)
        heights.append(h)

        # NaNs
        nan_count = spec_df.isna().sum().sum()
        nan_ratios.append(nan_count / spec_df.size)

        # Stats
        vals = spec_df.values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            means.append(np.mean(vals))
            stds.append(np.std(vals))

    if widths:
        print(f"Dimension Distribution:")
        print(
            f"  Width (Freq Bins) : Mean={np.mean(widths):.1f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"  Height (Time Steps): Mean={np.mean(heights):.1f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(f"Global Pixel Mean   : {np.mean(means):.4f}")
        print(f"Global Pixel Std    : {np.mean(stds):.4f}")
        print(f"Average NaN Ratio   : {np.mean(nan_ratios)*100:.4f}%")

    return pd.DataFrame(
        {
            "spectrogram_path": unique_spec_paths.values,
            "spec_mean": means,
            "spec_std": stds,
            "spec_nan_ratio": nan_ratios,
        }
    )


def analyze_relationships(df, eeg_stats, spec_stats):
    print_section("Feature & Signal Relationships")

    # Merge extracted stats back to the main dataframe
    # Note: df has multiple rows per file (different sub_ids), but stats are per file.
    # We will merge and drop duplicates to analyze file-level relationships.

    # Prepare target column: 'expert_consensus' encoded
    le = LabelEncoder()
    df["target_encoded"] = le.fit_transform(df["expert_consensus"])

    # 1. Metadata vs Target
    # Does the number of votes correlate with the "confidence" (max probability)?
    df["max_prob"] = df[
        ["seizure_prob", "lpd_prob", "gpd_prob", "lrda_prob", "grda_prob", "other_prob"]
    ].max(axis=1)
    corr_votes_conf = df["total_votes"].corr(df["max_prob"])
    print(f"Correlation (Total Votes vs. Max Probability): {corr_votes_conf:.4f}")

    # 2. Signal Stats vs Target (using the sampled subset)
    # Merge EEG stats
    if eeg_stats is not None and not eeg_stats.empty:
        merged_eeg = df.merge(eeg_stats, on="eeg_path", how="inner")
        # Drop duplicates to avoid weighting files with more sub-samples higher
        merged_eeg_unique = merged_eeg.drop_duplicates(subset=["eeg_path"])

        print("\nEEG Signal Stats vs Target Confidence (Correlation):")
        print(
            f"  Signal Mean vs Max Prob: {merged_eeg_unique['eeg_mean'].corr(merged_eeg_unique['max_prob']):.4f}"
        )
        print(
            f"  Signal Std  vs Max Prob: {merged_eeg_unique['eeg_std'].corr(merged_eeg_unique['max_prob']):.4f}"
        )
        print(
            f"  NaN Ratio   vs Max Prob: {merged_eeg_unique['eeg_nan_ratio'].corr(merged_eeg_unique['max_prob']):.4f}"
        )

    # 3. Feature Importance (Tabular + Meta-features)
    # We will build a small dataset from the sampled files to check importance
    # We need rows where we have both EEG and Spec stats if possible, or just use what we have.
    # Let's use the full DF for metadata importance first.

    print("\nFeature Importance (Random Forest on Metadata):")
    features = [
        "total_votes",
        "eeg_label_offset_seconds",
        "spectogram_label_offset_seconds",
    ]
    features = [f for f in features if f in df.columns]

    # Drop NaNs for RF
    rf_df = df[features + ["target_encoded"]].dropna()

    if not rf_df.empty:
        X = rf_df[features]
        y = rf_df["target_encoded"]

        rf = RandomForestRegressor(
            n_estimators=50, random_state=RANDOM_STATE, verbose=0
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=features).sort_values(
            ascending=False
        )
        print("Top Metadata Features predicting Expert Consensus:")
        for name, imp in importances.head(5).items():
            print(f"  {name:<30}: {imp:.4f}")
    else:
        print("  Insufficient data for Random Forest analysis.")


def main():
    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    print(
        f"Loaded Training Metadata: {len(df)} rows, {df['patient_id'].nunique()} patients."
    )

    # 2. Target Analysis
    analyze_targets(df)

    # 3. Tabular Input Analysis
    analyze_tabular_metadata(df)

    # 4. Modality Specific Analysis (with Sampling)
    eeg_stats = analyze_eeg_data(df)
    spec_stats = analyze_spectrogram_data(df)

    # 5. Relationships
    analyze_relationships(df, eeg_stats, spec_stats)


if __name__ == "__main__":
    main()
