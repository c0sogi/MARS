import os
import glob
import random
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis

# ==========================================
# Configuration & Setup
# ==========================================
# Suppress warnings and progress bars
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")

# Constants
SAMPLE_SIZE_FILES = 200  # Number of files to sample for heavy I/O checks
RF_SAMPLE_SIZE = 5000  # Number of rows to use for lightweight RF training


def print_section(title):
    print(f"\n{'='*40}")
    print(f"{title}")
    print(f"{'='*40}")


def analyze_targets(df):
    print_section("TARGET VARIABLE ANALYSIS")

    # Target columns (Probabilities)
    target_cols = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]

    # 1. Distribution of Probabilities (Mean across dataset)
    print("Global Mean Probabilities per Class:")
    means = df[target_cols].mean()
    for col, val in means.items():
        print(f"  {col:<15}: {val:.4f}")

    # 2. Hard Classification Balance (Argmax)
    # We convert the soft probs to a hard label for balance analysis
    df["hard_label"] = df[target_cols].idxmax(axis=1)
    class_counts = df["hard_label"].value_counts()
    total_samples = len(df)

    print("\nClass Balance (based on dominant probability):")
    for label, count in class_counts.items():
        ratio = count / total_samples
        print(f"  {label:<15}: {count} ({ratio:.2%})")

    # 3. Check for Uniform Distributions (High Uncertainty)
    # If all probs are roughly 1/6 (0.1667), it implies total disagreement or lack of signal
    # We check if the max prob is close to 1/6
    low_confidence_threshold = 0.20
    low_conf_count = (df[target_cols].max(axis=1) < low_confidence_threshold).sum()
    print(
        f"\nLow Confidence Samples (Max Prob < {low_confidence_threshold}): {low_conf_count} ({low_conf_count/total_samples:.2%})"
    )


def analyze_tabular_metadata(df):
    print_section("TABULAR DATA ANALYSIS (METADATA)")

    # Features to analyze (excluding targets and paths)
    num_cols = [
        "eeg_label_offset_seconds",
        "spectogram_label_offset_seconds",
        "total_votes",
    ]
    cat_cols = [
        "patient_id",
        "expert_consensus",
    ]  # expert_consensus is in original train.csv, let's check if it's in metadata

    # Check which columns actually exist in the loaded dataframe
    existing_num = [c for c in num_cols if c in df.columns]
    existing_cat = [c for c in cat_cols if c in df.columns]

    # 1. Numerical Analysis
    print("Numerical Columns Statistics:")
    if existing_num:
        stats = df[existing_num].describe().T
        for idx, row in stats.iterrows():
            # Outlier detection (IQR)
            Q1 = df[idx].quantile(0.25)
            Q3 = df[idx].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (
                (df[idx] < (Q1 - 1.5 * IQR)) | (df[idx] > (Q3 + 1.5 * IQR))
            ).sum()

            print(f"  {idx}:")
            print(f"    Mean: {row['mean']:.4f}, Std: {row['std']:.4f}")
            print(f"    Min: {row['min']:.4f}, Max: {row['max']:.4f}")
            print(f"    Outliers (IQR method): {outliers}")
    else:
        print("  No numerical metadata columns found.")

    # 2. Categorical Analysis
    print("\nCategorical Columns Statistics:")
    for col in existing_cat:
        unique_vals = df[col].nunique()
        print(f"  {col}: {unique_vals} unique values")
        if unique_vals > 50:
            print(
                f"    High cardinality column (Top 5): {df[col].value_counts().head(5).index.tolist()}"
            )
        else:
            # Check for rare labels
            counts = df[col].value_counts(normalize=True)
            rare = counts[counts < 0.01]
            if not rare.empty:
                print(f"    Rare labels (<1%): {rare.index.tolist()}")
            else:
                print("    No rare labels found.")

    # 3. Missing Values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(df):.2%})")
    else:
        print("  No missing values in metadata.")


def analyze_eeg_data(df):
    print_section("EEG DATA ANALYSIS (SIGNAL)")

    # Sample files
    sample_paths = (
        df["eeg_path"]
        .sample(n=min(SAMPLE_SIZE_FILES, len(df)), random_state=SEED)
        .tolist()
    )
    full_paths = [os.path.join(INPUT_DIR, p) for p in sample_paths]

    durations = []
    channel_counts = []
    global_min = float("inf")
    global_max = float("-inf")
    nan_counts = []
    sampling_rates = []  # Inferred

    print(f"Analyzing a sample of {len(full_paths)} EEG parquet files...")

    for p in full_paths:
        try:
            # Read parquet
            eeg_df = pd.read_parquet(p)

            # Dimensions
            time_steps, n_channels = eeg_df.shape
            channel_counts.append(n_channels)

            # Duration & Sampling Rate
            # Known: 200 Hz. Duration = time_steps / 200
            sr = 200
            durations.append(time_steps / sr)
            sampling_rates.append(sr)

            # Values
            # We use numpy for speed
            vals = eeg_df.values

            # NaNs
            n_nans = np.isnan(vals).sum()
            nan_counts.append(n_nans / vals.size)

            # Min/Max (ignoring NaNs)
            if n_nans < vals.size:
                file_min = np.nanmin(vals)
                file_max = np.nanmax(vals)
                if file_min < global_min:
                    global_min = file_min
                if file_max > global_max:
                    global_max = file_max

        except Exception as e:
            continue

    # Report
    if durations:
        print(
            f"  Duration (seconds): Mean={np.mean(durations):.4f}, Std={np.std(durations):.4f}"
        )
        print(f"  Channel Counts: Unique values found: {np.unique(channel_counts)}")
        print(f"  Sampling Rates: Unique values found: {np.unique(sampling_rates)}")
        print(
            f"  Global Pixel/Signal Stats: Min={global_min:.4f}, Max={global_max:.4f}"
        )
        print(
            f"  NaN Ratio per file: Mean={np.mean(nan_counts):.4f}, Max={np.max(nan_counts):.4f}"
        )

        if np.mean(nan_counts) > 0:
            print(
                "  NOTE: EEG data contains NaNs. Preprocessing must handle missing signal values."
            )
    else:
        print("  Could not load EEG files.")


def analyze_spectrogram_data(df):
    print_section("SPECTROGRAM DATA ANALYSIS (IMAGE)")

    # Sample files
    sample_paths = (
        df["spec_path"]
        .sample(n=min(SAMPLE_SIZE_FILES, len(df)), random_state=SEED)
        .tolist()
    )
    full_paths = [os.path.join(INPUT_DIR, p) for p in sample_paths]

    widths = []  # Time
    heights = []  # Frequencies
    means = []
    stds = []
    nan_ratios = []

    print(f"Analyzing a sample of {len(full_paths)} Spectrogram parquet files...")

    for p in full_paths:
        try:
            spec_df = pd.read_parquet(p)

            # The parquet columns are frequencies, rows are time
            # Height = number of columns (frequencies), Width = number of rows (time)
            # Note: Usually spectrograms are (Freq, Time), but dataframes are (Time, Freq)
            h, w = spec_df.shape[1], spec_df.shape[0]
            widths.append(w)
            heights.append(h)

            vals = spec_df.values

            # NaNs
            n_nans = np.isnan(vals).sum()
            nan_ratios.append(n_nans / vals.size)

            # Stats
            if n_nans < vals.size:
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))

        except Exception as e:
            continue

    if widths:
        print(
            f"  Dimensions (Time steps): Mean={np.mean(widths):.4f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )
        print(
            f"  Dimensions (Frequency bins): Mean={np.mean(heights):.4f}, Unique={np.unique(heights)}"
        )
        print(f"  Pixel Value Mean: {np.mean(means):.4f}")
        print(f"  Pixel Value Std: {np.mean(stds):.4f}")
        print(f"  NaN Ratio per file: Mean={np.mean(nan_ratios):.4f}")

        if np.mean(nan_ratios) > 0:
            print("  NOTE: Spectrograms contain NaNs. Imputation or masking required.")
    else:
        print("  Could not load Spectrogram files.")


def analyze_relationships(df):
    print_section("FEATURE/SIGNAL RELATIONSHIPS")

    # 1. Correlation Matrix (Numerical Metadata vs Targets)
    # We define targets as the probability columns
    target_cols = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]
    meta_cols = [
        "eeg_label_offset_seconds",
        "spectogram_label_offset_seconds",
        "total_votes",
    ]

    # Filter to existing
    meta_cols = [c for c in meta_cols if c in df.columns]

    if meta_cols:
        print("Correlation (Pearson) between Metadata and Target Probabilities:")
        # Compute correlation
        corr_df = df[meta_cols + target_cols].corr()
        # Extract just the Meta vs Target block
        subset_corr = corr_df.loc[meta_cols, target_cols]
        print(subset_corr.round(4))

        # Check for redundancy among metadata
        print("\nRedundancy Check (Metadata Correlation > 0.9):")
        meta_corr = df[meta_cols].corr().abs()
        # Upper triangle
        upper = meta_corr.where(np.triu(np.ones(meta_corr.shape), k=1).astype(bool))
        high_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
        if high_corr:
            print(f"  High collinearity found in: {high_corr}")
        else:
            print("  No highly collinear metadata features found.")

    # 2. Feature Importance (Lightweight RF)
    # We try to predict the 'hard_label' using available metadata
    print("\nFeature Importance (Random Forest on Metadata):")

    # Prepare data
    if "hard_label" not in df.columns:
        df["hard_label"] = df[target_cols].idxmax(axis=1)

    # Sample data for speed
    sample_df = df.sample(n=min(RF_SAMPLE_SIZE, len(df)), random_state=SEED).copy()

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(sample_df["hard_label"])

    # Prepare X
    # We use numerical cols + patient_id (encoded)
    X = sample_df[meta_cols].fillna(0)

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    # Get importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("  Top Features predicting dominant class:")
    for f in range(min(5, len(meta_cols))):
        print(f"    {meta_cols[indices[f]]}: {importances[indices[f]]:.4f}")


def main():
    # Load Data
    if not os.path.exists(TRAIN_META_PATH):
        print(f"Error: Metadata file not found at {TRAIN_META_PATH}")
        return

    df_train = pd.read_csv(TRAIN_META_PATH)

    # Run Analysis Modules
    analyze_targets(df_train)
    analyze_tabular_metadata(df_train)
    analyze_eeg_data(df_train)
    analyze_spectrogram_data(df_train)
    analyze_relationships(df_train)

    print("\nEDA Completed Successfully.")


if __name__ == "__main__":
    main()
