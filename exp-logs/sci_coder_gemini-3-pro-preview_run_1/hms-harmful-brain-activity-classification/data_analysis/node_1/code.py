import os
import numpy as np
import pandas as pd
import glob
import random
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")

SEED = 42
SAMPLE_SIZE = 100  # Number of files to sample for heavy IO operations

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_section(title):
    print(f"\n{'='*10} {title.upper()} {'='*10}")


def analyze_targets(df):
    print_section("Target Variable Analysis")

    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    print(f"Target Columns (Probabilities): {prob_cols}")

    # Distribution of Probabilities
    print("\n--- Class Probability Distribution (Mean) ---")
    means = df[prob_cols].mean().sort_values(ascending=False)
    for col, val in means.items():
        print(f"{col}: {val:.4f}")

    # Hard Label Analysis (Argmax)
    print("\n--- Dominant Class Distribution (Hard Labels) ---")
    # Map prob cols to class names
    class_names = [c.replace("_prob", "") for c in prob_cols]
    hard_labels = df[prob_cols].idxmax(axis=1).apply(lambda x: x.replace("_prob", ""))

    counts = hard_labels.value_counts()
    total = len(df)
    for label, count in counts.items():
        ratio = count / total
        print(f"{label}: {count} ({ratio:.4f})")

    # Check for sum constraints
    sums = df[prob_cols].sum(axis=1)
    invalid_sums = sums[~np.isclose(sums, 1.0, atol=1e-4)]
    if len(invalid_sums) > 0:
        print(f"\nWARNING: {len(invalid_sums)} rows do not sum to 1.0")
    else:
        print("\nIntegrity Check: All target probability rows sum to 1.0")


def analyze_tabular(df):
    print_section("Tabular Data Analysis (Metadata)")

    # Numerical Columns
    num_cols = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]
    print("\n--- Numerical Metadata Stats ---")
    for col in num_cols:
        if col in df.columns:
            stats = df[col].describe()
            iqr = stats["75%"] - stats["25%"]
            lower = stats["25%"] - 1.5 * iqr
            upper = stats["75%"] + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]

            print(f"Column: {col}")
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
            print(f"  Outliers (IQR method): {outliers}")

    # Categorical Columns
    cat_cols = ["expert_consensus", "patient_id"]
    print("\n--- Categorical Metadata Stats ---")
    for col in cat_cols:
        if col in df.columns:
            unique_vals = df[col].nunique()
            print(f"Column: {col} | Cardinality: {unique_vals}")
            if unique_vals < 50:
                print(
                    f"  Distribution: {df[col].value_counts(normalize=True).to_dict()}"
                )

    # Missing Values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("No missing values found in metadata.")
    else:
        for col, count in missing.items():
            print(f"{col}: {count} ({count/len(df):.4f})")


def analyze_eeg_signals(df):
    print_section("Audio/Signal Data Analysis (EEG)")

    # Sample files
    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)

    durations = []
    shapes = []
    nan_counts = []
    means = []
    stds = []

    print(f"Analyzing {len(sample_df)} sampled EEG files...")

    for _, row in sample_df.iterrows():
        path = os.path.join(INPUT_DIR, row["eeg_path"])
        try:
            # Read parquet
            eeg_data = pd.read_parquet(path)

            # Dimensions
            shapes.append(eeg_data.shape)
            # Assuming 200Hz, duration = rows / 200
            durations.append(eeg_data.shape[0] / 200.0)

            # NaNs
            nans = eeg_data.isna().sum().sum()
            nan_counts.append(nans)

            # Stats (Global for this file)
            # Fill NaNs for stats calculation to avoid errors
            vals = eeg_data.values
            # Simple mean/std ignoring NaNs
            means.append(np.nanmean(vals))
            stds.append(np.nanstd(vals))

        except Exception as e:
            print(f"Error reading {path}: {e}")

    # Report
    if shapes:
        avg_rows = np.mean([s[0] for s in shapes])
        avg_cols = np.mean([s[1] for s in shapes])
        print(
            f"Average Dimensions: {avg_rows:.1f} rows (time), {avg_cols:.1f} columns (channels)"
        )
        print(f"Average Duration: {np.mean(durations):.4f} seconds")
        print(
            f"Sampling Rate Check: {avg_rows/np.mean(durations):.1f} Hz (Expected ~200.0)"
        )

        # Check for mono/stereo inconsistency (channel counts)
        unique_channels = set([s[1] for s in shapes])
        print(f"Unique Channel Counts found: {unique_channels}")

        # NaNs
        avg_nan_ratio = np.mean(nan_counts) / (avg_rows * avg_cols)
        print(f"Average NaN Ratio per file: {avg_nan_ratio:.4f}")

        # Signal Stats
        print(f"Global Signal Mean: {np.mean(means):.4f}")
        print(f"Global Signal Std: {np.mean(stds):.4f}")


def analyze_spectrograms(df):
    print_section("Image Data Analysis (Spectrograms)")

    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)

    widths = []
    heights = []
    nan_counts = []
    means = []
    stds = []

    print(f"Analyzing {len(sample_df)} sampled Spectrogram files...")

    for _, row in sample_df.iterrows():
        path = os.path.join(INPUT_DIR, row["spectrogram_path"])
        try:
            spec_data = pd.read_parquet(path)

            # Dimensions: Rows=Time, Cols=Frequency
            # In image terms, Height=Time, Width=Freq (or vice versa depending on interpretation)
            # Here we just report raw shape
            h, w = spec_data.shape
            heights.append(h)
            widths.append(w)

            # NaNs
            nans = spec_data.isna().sum().sum()
            nan_counts.append(nans)

            # Stats
            vals = spec_data.values
            means.append(np.nanmean(vals))
            stds.append(np.nanstd(vals))

        except Exception as e:
            print(f"Error reading {path}: {e}")

    if widths:
        print(
            f"Height Distribution (Time steps): Mean={np.mean(heights):.1f}, Min={np.min(heights)}, Max={np.max(heights)}"
        )
        print(
            f"Width Distribution (Freq bins): Mean={np.mean(widths):.1f}, Min={np.min(widths)}, Max={np.max(widths)}"
        )

        unique_w = set(widths)
        if len(unique_w) == 1:
            print("Consistent width (frequency resolution) across samples.")
        else:
            print(f"Inconsistent widths found: {unique_w}")

        avg_nan_ratio = np.mean(nan_counts) / (np.mean(heights) * np.mean(widths))
        print(f"Average NaN Ratio per file: {avg_nan_ratio:.4f}")

        print(f"Global Pixel Mean: {np.mean(means):.4f}")
        print(f"Global Pixel Std: {np.mean(stds):.4f}")


def analyze_relationships(df):
    print_section("Feature/Signal Relationships")

    prob_cols = [c for c in df.columns if c.endswith("_prob")]

    # 1. Correlation between targets
    print("\n--- Target Correlation (Pearson) ---")
    corr = df[prob_cols].corr()
    # Print pairs with high correlation (absolute > 0.5)
    high_corr_found = False
    for i in range(len(prob_cols)):
        for j in range(i + 1, len(prob_cols)):
            val = corr.iloc[i, j]
            if abs(val) > 0.5:
                print(f"{prob_cols[i]} vs {prob_cols[j]}: {val:.4f}")
                high_corr_found = True
    if not high_corr_found:
        print("No strong correlations (>0.5) between target probabilities.")

    # 2. Metadata vs Target (Random Forest Importance)
    print("\n--- Metadata Importance (Random Forest) ---")
    # Prepare data
    feature_cols = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]
    # Encode expert_consensus for target
    le = LabelEncoder()
    y = le.fit_transform(df["expert_consensus"])

    X = df[feature_cols].fillna(0)

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top Features predicting 'expert_consensus':")
    for f in range(len(feature_cols)):
        idx = indices[f]
        print(f"{f+1}. {feature_cols[idx]}: {importances[idx]:.4f}")

    # 3. Offset vs Seizure Probability
    # Check correlation between offset and seizure probability
    if "seizure_prob" in df.columns and "eeg_label_offset_seconds" in df.columns:
        corr_offset_seizure = df["eeg_label_offset_seconds"].corr(df["seizure_prob"])
        print(
            f"\nCorrelation between EEG Offset and Seizure Probability: {corr_offset_seizure:.4f}"
        )


def main():
    set_seed(SEED)

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    print("Loading metadata...")
    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(df)} rows.")

    analyze_targets(df)
    analyze_tabular(df)
    analyze_eeg_signals(df)
    analyze_spectrograms(df)
    analyze_relationships(df)

    print_section("EDA Complete")


if __name__ == "__main__":
    main()
