import os
import pandas as pd
import numpy as np
import soundfile as sf
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target_variable(df):
    print("=== TARGET VARIABLE ANALYSIS ===")

    # Distribution of labels
    label_counts = df["label"].value_counts()
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {len(label_counts)}")

    print("\nClass Distribution:")
    for label, count in label_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {label:<10}: {count} ({percentage:.4f}%)")

    # Imbalance check
    max_class_count = label_counts.max()
    min_class_count = label_counts.min()
    imbalance_ratio = (
        max_class_count / min_class_count if min_class_count > 0 else float("inf")
    )

    print(f"\nClass Balance Ratio (Max/Min): {imbalance_ratio:.4f}")
    if imbalance_ratio > 10:
        print("  -> Significant class imbalance detected (Ratio > 10).")
    elif imbalance_ratio > 2:
        print("  -> Moderate class imbalance detected (Ratio > 2).")
    else:
        print("  -> Classes are relatively balanced.")


def analyze_audio_data(df, input_dir):
    print("\n=== INPUT DATA ANALYSIS (AUDIO) ===")

    durations = []
    sample_rates = []
    channels = []
    subtypes = []  # Bit depth proxy

    # We will process all files. sf.info is fast (header read only).
    # Construct full paths
    # Metadata paths are relative, e.g., train/audio/bed/file.wav
    # Input dir is ./input

    print(f"Processing {len(df)} audio files for signal analysis...")

    valid_indices = []

    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        try:
            info = sf.info(full_path)
            durations.append(info.duration)
            sample_rates.append(info.samplerate)
            channels.append(info.channels)
            subtypes.append(info.subtype)
            valid_indices.append(idx)
        except Exception as e:
            # Silent fail for individual file errors, just skip
            continue

    # Convert to numpy arrays for stats
    durations = np.array(durations)
    sample_rates = np.array(sample_rates)
    channels = np.array(channels)

    # 1. Signal Duration
    print("\n--- Duration Statistics (Seconds) ---")
    print(f"Mean: {np.mean(durations):.4f}")
    print(f"Std : {np.std(durations):.4f}")
    print(f"Min : {np.min(durations):.4f}")
    print(f"Max : {np.max(durations):.4f}")

    # outlier check (IQR)
    q1 = np.percentile(durations, 25)
    q3 = np.percentile(durations, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.sum((durations < lower_bound) | (durations > upper_bound))
    print(f"Outliers (IQR method): {outliers} ({outliers/len(durations)*100:.2f}%)")

    # 2. Sampling Rates
    print("\n--- Sampling Rate Distribution ---")
    unique_sr, counts_sr = np.unique(sample_rates, return_counts=True)
    for sr, count in zip(unique_sr, counts_sr):
        print(f"  {sr} Hz: {count} files ({count/len(sample_rates)*100:.2f}%)")

    # 3. Channels
    print("\n--- Channel Distribution ---")
    unique_ch, counts_ch = np.unique(channels, return_counts=True)
    for ch, count in zip(unique_ch, counts_ch):
        type_str = "Mono" if ch == 1 else "Stereo" if ch == 2 else f"{ch}-channel"
        print(f"  {ch} ({type_str}): {count} files ({count/len(channels)*100:.2f}%)")

    # 4. Bit Depth (Subtype)
    print("\n--- Bit Depth / Subtype Distribution ---")
    unique_sub, counts_sub = np.unique(subtypes, return_counts=True)
    for sub, count in zip(unique_sub, counts_sub):
        print(f"  {sub}: {count} files ({count/len(subtypes)*100:.2f}%)")

    # Return gathered data for relationship analysis
    # Filter df to only valid indices
    df_valid = df.loc[valid_indices].copy()
    df_valid["duration"] = durations
    df_valid["sample_rate"] = sample_rates
    return df_valid


def analyze_relationships(df):
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # 1. Duration vs Label
    print("\n--- Duration vs Target Label ---")
    print("Do specific classes have distinct duration profiles?")

    # Group by label and calculate duration stats
    duration_stats = df.groupby("label")["duration"].agg(
        ["mean", "std", "min", "max", "count"]
    )
    duration_stats = duration_stats.sort_values(by="mean", ascending=False)

    print(
        f"{'Label':<10} | {'Mean (s)':<10} | {'Std (s)':<10} | {'Min (s)':<10} | {'Max (s)':<10}"
    )
    print("-" * 60)
    for label, row in duration_stats.iterrows():
        print(
            f"{label:<10} | {row['mean']:<10.4f} | {row['std']:<10.4f} | {row['min']:<10.4f} | {row['max']:<10.4f}"
        )

    # Check for correlation between duration and label (using one-hot or just observation)
    # Since label is categorical, we look for variance between groups.
    # We can highlight if "silence" (background noise) is significantly different.

    if "silence" in duration_stats.index:
        silence_mean = duration_stats.loc["silence", "mean"]
        others_mean = duration_stats.loc[
            duration_stats.index != "silence", "mean"
        ].mean()
        print(
            f"\nObservation: 'silence' mean duration ({silence_mean:.4f}s) vs Others mean ({others_mean:.4f}s)."
        )

    # 2. Sample Rate Consistency across Labels
    print("\n--- Sample Rate Consistency ---")
    # Check if any label has mixed sample rates
    mixed_sr_labels = []
    for label, group in df.groupby("label"):
        unique_srs = group["sample_rate"].unique()
        if len(unique_srs) > 1:
            mixed_sr_labels.append((label, unique_srs))

    if not mixed_sr_labels:
        print("All labels have consistent sampling rates internally.")
    else:
        print("Labels with mixed sampling rates:")
        for label, srs in mixed_sr_labels:
            print(f"  {label}: {srs}")


def main():
    # 1. Setup
    set_seed(42)
    INPUT_DIR = "./input"
    METADATA_FILE = "./metadata/train.csv"

    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    # 2. Load Data
    try:
        df_train = pd.read_csv(METADATA_FILE)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # 3. Target Variable Analysis
    analyze_target_variable(df_train)

    # 4. Input Data Analysis (Audio)
    # This returns the dataframe augmented with audio stats
    df_augmented = analyze_audio_data(df_train, INPUT_DIR)

    # 5. Feature/Signal Relationships
    analyze_relationships(df_augmented)

    print("\nEDA Completed Successfully.")


if __name__ == "__main__":
    main()
