import os
import numpy as np
import pandas as pd
import random
import sys

# --- Constants ---
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train_metadata.csv"
SAMPLE_SIZE = 2000  # Number of files to analyze for pixel stats
RANDOM_SEED = 42


# --- Setup ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        sys.exit(1)
    return pd.read_csv(METADATA_PATH)


# --- Analysis Functions ---


def analyze_target(df):
    print("TARGET VARIABLE ANALYSIS")
    print("-" * 30)

    counts = df["target"].value_counts()
    proportions = df["target"].value_counts(normalize=True)

    print(f"Target Distribution:")
    for label, count in counts.items():
        prop = proportions[label]
        print(f"  Class {label}: {count} samples ({prop:.4%})")

    # Imbalance
    if len(counts) == 2:
        ratio = counts[0] / counts[1] if counts[1] > 0 else 0
        print(f"Class Imbalance Ratio (0:1): {ratio:.4f} : 1")
    else:
        print("  Data does not contain both binary classes.")
    print("")


def analyze_images(df):
    print("INPUT DATA ANALYSIS (SPECTROGRAM/IMAGE)")
    print("-" * 30)

    # Stratified Sample
    try:
        # Attempt stratified sampling
        sample_df = df.groupby("target", group_keys=False).apply(
            lambda x: x.sample(min(len(x), SAMPLE_SIZE // 2), random_state=RANDOM_SEED)
        )
        # If we didn't get enough (e.g. one class is too small), fill up with random
        if len(sample_df) < SAMPLE_SIZE and len(df) > len(sample_df):
            remaining = df.drop(sample_df.index)
            n_needed = min(len(remaining), SAMPLE_SIZE - len(sample_df))
            extra = remaining.sample(n=n_needed, random_state=RANDOM_SEED)
            sample_df = pd.concat([sample_df, extra])
    except Exception:
        # Fallback to simple random sample
        sample_df = df.sample(n=min(len(df), SAMPLE_SIZE), random_state=RANDOM_SEED)

    print(f"Analyzing a sample of {len(sample_df)} files for pixel statistics...")

    # Accumulators for Welford's online algorithm or simple sum/sq_sum
    # Using float64 to prevent overflow
    total_pixels = 0
    sum_val = 0.0
    sum_sq_val = 0.0
    min_val = float("inf")
    max_val = float("-inf")

    dims_consistent = True
    expected_shape = (6, 273, 256)

    # Lists to store meta-features for relationship analysis
    meta_features = []

    for _, row in sample_df.iterrows():
        file_rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, file_rel_path)

        try:
            # Load data
            # Data is float16, convert to float32 for calculations
            img = np.load(full_path).astype(np.float32)

            # Check Dimensions
            if img.shape != expected_shape:
                dims_consistent = False

            # Global Stats
            current_min = np.min(img)
            current_max = np.max(img)

            if current_min < min_val:
                min_val = current_min
            if current_max > max_val:
                max_val = current_max

            sum_val += np.sum(img)
            sum_sq_val += np.sum(img**2)
            total_pixels += img.size

            # Meta-Feature Extraction
            # Panels 0, 2, 4 are "ON" target (A)
            # Panels 1, 3, 5 are "OFF" target (B, C, D)
            on_target = img[[0, 2, 4], :, :]
            off_target = img[[1, 3, 5], :, :]

            mean_on = np.mean(on_target)
            mean_off = np.mean(off_target)
            std_on = np.std(on_target)
            max_on = np.max(on_target)

            meta_features.append(
                {
                    "target": row["target"],
                    "mean_on": mean_on,
                    "mean_off": mean_off,
                    "std_on": std_on,
                    "max_on": max_on,
                    "contrast": mean_on - mean_off,
                }
            )

        except Exception as e:
            # In a real scenario we might log this, but for EDA script we skip
            continue

    # Calculate Global Stats
    global_mean = sum_val / total_pixels
    global_std = np.sqrt((sum_sq_val / total_pixels) - (global_mean**2))

    print(f"Dimensions: {expected_shape} (Consistent: {dims_consistent})")
    print(f"Channel Count: {expected_shape[0]} (6 'cadence' panels)")
    print(f"Global Pixel Mean: {global_mean:.4f}")
    print(f"Global Pixel Std:  {global_std:.4f}")
    print(f"Global Pixel Min:  {min_val:.4f}")
    print(f"Global Pixel Max:  {max_val:.4f}")
    print("")

    return pd.DataFrame(meta_features)


def analyze_relationships(meta_df):
    print("FEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 30)

    if meta_df.empty:
        print("No data available for relationship analysis.")
        return

    # 1. Correlation Analysis
    # We look at how signal properties correlate with the target
    correlations = meta_df.corr()["target"].drop("target")

    print("Correlation with Target (Point-Biserial):")
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.4f}")

    print("\nFeature Importance (Proxy via Mean Difference):")
    # Compare means of features for Target 0 vs Target 1
    grouped = meta_df.groupby("target").mean()

    for col in grouped.columns:
        val_0 = grouped.loc[0, col] if 0 in grouped.index else np.nan
        val_1 = grouped.loc[1, col] if 1 in grouped.index else np.nan
        diff = val_1 - val_0
        print(
            f"  {col}: Class 0 Mean = {val_0:.4f}, Class 1 Mean = {val_1:.4f}, Diff = {diff:.4f}"
        )

    print("\nObservation:")
    print(
        "  'mean_on' and 'contrast' differences suggest whether energy in 'A' panels distinguishes targets."
    )


def main():
    set_seed(RANDOM_SEED)

    # 1. Load Metadata
    df_train = load_metadata()

    # 2. Target Analysis
    analyze_target(df_train)

    # 3. Input Data Analysis (and collect meta-features)
    meta_df = analyze_images(df_train)

    # 4. Feature Relationships
    analyze_relationships(meta_df)


if __name__ == "__main__":
    main()
