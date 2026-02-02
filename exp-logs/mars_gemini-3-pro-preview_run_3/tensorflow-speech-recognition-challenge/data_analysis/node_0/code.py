import os
import pandas as pd
import numpy as np
import soundfile as sf
import random
import warnings

# Configuration
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SEED = 42

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target_variable(df):
    print("=== TARGET VARIABLE ANALYSIS ===")

    # Distribution
    counts = df["label"].value_counts()
    total = len(df)

    print(f"Total Samples: {total}")
    print(f"Number of Classes: {len(counts)}")
    print("\nClass Distribution:")
    for label, count in counts.items():
        percent = (count / total) * 100
        print(f"  {label:<10}: {count} ({percent:.4f}%)")

    # Imbalance
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else 0

    print(f"\nClass Imbalance Ratio (Most/Least Frequent): {imbalance_ratio:.4f}")
    if imbalance_ratio > 10:
        print("  -> Significant imbalance detected.")
    elif imbalance_ratio > 2:
        print("  -> Moderate imbalance detected.")
    else:
        print("  -> Dataset is relatively balanced.")


def get_audio_info(filepath):
    """Extracts metadata from a single audio file without loading the whole array."""
    full_path = os.path.join(INPUT_DIR, filepath)
    try:
        info = sf.info(full_path)
        return {
            "duration": info.duration,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,  # Proxy for bit depth (e.g., PCM_16)
        }
    except Exception as e:
        return None


def analyze_audio_data(df):
    print("\n=== AUDIO DATA ANALYSIS ===")

    # 1. Metadata Extraction (Duration, SR, Channels)
    # We process all files for metadata as sf.info is fast
    print("Extracting audio metadata from files...")

    meta_records = []
    # Using a list comprehension or loop is fine here.
    # Since we need to join back to df for relationship analysis, we'll keep order.
    for idx, row in df.iterrows():
        info = get_audio_info(row["filepath"])
        if info:
            meta_records.append(info)
        else:
            meta_records.append(
                {
                    "duration": np.nan,
                    "samplerate": np.nan,
                    "channels": np.nan,
                    "subtype": "ERROR",
                }
            )

    df_meta = pd.DataFrame(meta_records)
    df_combined = pd.concat([df, df_meta], axis=1)

    # Drop failures if any
    df_clean = df_combined.dropna(subset=["duration"])

    # Duration Analysis
    durations = df_clean["duration"]
    print(f"\nDuration (seconds):")
    print(f"  Mean: {durations.mean():.4f}")
    print(f"  Std : {durations.std():.4f}")
    print(f"  Min : {durations.min():.4f}")
    print(f"  Max : {durations.max():.4f}")

    # Sampling Rate Analysis
    sr_counts = df_clean["samplerate"].value_counts()
    print(f"\nSampling Rates:")
    for sr, count in sr_counts.items():
        print(f"  {sr} Hz: {count} files")

    # Channel Analysis
    ch_counts = df_clean["channels"].value_counts()
    print(f"\nChannels:")
    for ch, count in ch_counts.items():
        type_str = "Mono" if ch == 1 else "Stereo" if ch == 2 else "Multi"
        print(f"  {ch} ({type_str}): {count} files")

    # Bit Depth / Subtype Analysis
    bd_counts = df_clean["subtype"].value_counts()
    print(f"\nBit Depth / Encoding:")
    for bd, count in bd_counts.items():
        print(f"  {bd}: {count} files")

    # 2. Signal Statistics (Global Mean/Std for Normalization)
    # We sample files to compute this to save time/memory
    sample_size = min(2000, len(df))
    print(
        f"\nCalculating global signal statistics on a sample of {sample_size} files..."
    )

    sample_indices = np.random.choice(df.index, size=sample_size, replace=False)

    accum_mean = 0.0
    accum_var = 0.0
    total_samples = 0

    # Welford's online algorithm or simple accumulation is tricky with variable lengths.
    # We will just concatenate a subset of raw data to compute stats.
    # Given memory constraints, we'll load them in batches or just list comprehension.

    all_samples = []
    for idx in sample_indices:
        fpath = os.path.join(INPUT_DIR, df.loc[idx, "filepath"])
        try:
            y, _ = sf.read(fpath)
            all_samples.append(y)
        except:
            pass

    if all_samples:
        flat_samples = np.concatenate(all_samples)
        global_mean = np.mean(flat_samples)
        global_std = np.std(flat_samples)

        print(f"  Global Signal Mean: {global_mean:.4f}")
        print(f"  Global Signal Std : {global_std:.4f}")
    else:
        print("  Could not compute signal stats.")

    return df_clean


def analyze_relationships(df):
    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # Relationship between Label and Duration
    # "Do longer audio files correlate with specific classes?"
    print("Relationship: Audio Duration vs Target Label")

    # Group by label
    grouped = df.groupby("label")["duration"].agg(["mean", "std", "count"])
    grouped = grouped.sort_values(by="mean", ascending=False)

    print(f"{'Label':<15} | {'Mean Dur (s)':<12} | {'Std Dur (s)':<12} | {'Count':<8}")
    print("-" * 55)
    for label, row in grouped.iterrows():
        print(
            f"{label:<15} | {row['mean']:<12.4f} | {row['std']:<12.4f} | {int(row['count']):<8}"
        )

    # Check for collinearity or redundancy in metadata?
    # In audio, SR and Channels are often constant.
    # We check if SR varies by label.
    print("\nRelationship: Sampling Rate consistency per Label")
    unique_srs_per_label = df.groupby("label")["samplerate"].nunique()
    if unique_srs_per_label.max() == 1:
        print("  All labels have consistent sampling rates within themselves.")
    else:
        print("  Some labels contain mixed sampling rates:")
        print(unique_srs_per_label[unique_srs_per_label > 1])


def main():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    df_train = pd.read_csv(METADATA_FILE)

    # 2. Target Analysis
    analyze_target_variable(df_train)

    # 3. Audio Analysis
    # This returns a DF with added metadata columns
    df_enriched = analyze_audio_data(df_train)

    # 4. Feature Relationships
    analyze_relationships(df_enriched)


if __name__ == "__main__":
    main()
