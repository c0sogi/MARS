import os
import pandas as pd
import numpy as np
import soundfile as sf
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_audio_metadata(row):
    """
    Reads audio header information.
    Returns a dictionary of metadata or None if read fails.
    """
    full_path = os.path.join(INPUT_DIR, row["filepath"])
    try:
        info = sf.info(full_path)
        return {
            "duration": info.duration,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,  # Proxy for bit depth
            "num_labels": len(row["labels"].split(",")),
        }
    except Exception as e:
        return None


def main():
    set_seed(SEED)

    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    print("========================================")
    print("      EXPLORATORY DATA ANALYSIS         ")
    print("========================================")

    # 2. Target Variable Analysis
    print("\nTARGET VARIABLE ANALYSIS")
    print("------------------------")

    # Parse labels
    all_labels = []
    label_counts_per_row = []

    for labels_str in df["labels"]:
        lbls = labels_str.split(",")
        all_labels.extend(lbls)
        label_counts_per_row.append(len(lbls))

    label_counts = Counter(all_labels)
    unique_labels = list(label_counts.keys())
    n_classes = len(unique_labels)

    # Distribution stats
    counts = list(label_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    mean_count = np.mean(counts)

    print(f"Task Type: Multi-Label Audio Classification")
    print(f"Total Samples: {len(df)}")
    print(f"Number of Unique Classes: {n_classes}")
    print(f"Average Labels per Clip: {np.mean(label_counts_per_row):.4f}")

    print(f"\nClass Balance:")
    print(
        f"  Max Class Frequency: {max_count} ({max(label_counts, key=label_counts.get)})"
    )
    print(
        f"  Min Class Frequency: {min_count} ({min(label_counts, key=label_counts.get)})"
    )
    print(f"  Mean Class Frequency: {mean_count:.4f}")
    print(f"  Imbalance Ratio (Max/Min): {max_count/min_count:.4f}")

    # Top 5 and Bottom 5
    sorted_labels = label_counts.most_common()
    print("\nTop 5 Most Frequent Classes:")
    for lbl, count in sorted_labels[:5]:
        print(f"  {lbl}: {count}")

    print("\nBottom 5 Least Frequent Classes:")
    for lbl, count in sorted_labels[-5:]:
        print(f"  {lbl}: {count}")

    # 3. Input Data Analysis (Audio Modality)
    print("\nINPUT DATA ANALYSIS (AUDIO)")
    print("---------------------------")

    # Extract audio metadata using threading for speed
    # We use a subset if the dataset is massive, but for ~20k files, reading headers is fast.
    print(f"Analyzing audio headers for {len(df)} files...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(get_audio_metadata, [row for _, row in df.iterrows()])
        )

    # Filter out Nones
    audio_meta = [r for r in results if r is not None]
    audio_df = pd.DataFrame(audio_meta)

    if len(audio_df) == 0:
        print("Failed to read audio files.")
        return

    # Signal Analysis: Duration
    durations = audio_df["duration"]
    print("\nSignal Properties (Duration in seconds):")
    print(f"  Mean: {durations.mean():.4f}")
    print(f"  Std : {durations.std():.4f}")
    print(f"  Min : {durations.min():.4f}")
    print(f"  25% : {durations.quantile(0.25):.4f}")
    print(f"  50% : {durations.median():.4f}")
    print(f"  75% : {durations.quantile(0.75):.4f}")
    print(f"  Max : {durations.max():.4f}")

    # Sampling Rates
    print("\nSampling Rates (Hz):")
    sr_counts = audio_df["samplerate"].value_counts()
    for sr, count in sr_counts.items():
        print(f"  {sr} Hz: {count} ({count/len(audio_df)*100:.2f}%)")

    # Bit Depths (Subtype)
    print("\nBit Depths / Subtypes:")
    bd_counts = audio_df["subtype"].value_counts()
    for bd, count in bd_counts.items():
        print(f"  {bd}: {count} ({count/len(audio_df)*100:.2f}%)")

    # Channels
    print("\nChannel Configuration:")
    ch_counts = audio_df["channels"].value_counts().sort_index()
    for ch, count in ch_counts.items():
        label = "Mono" if ch == 1 else "Stereo" if ch == 2 else f"{ch}-Channel"
        print(f"  {ch} ({label}): {count} ({count/len(audio_df)*100:.2f}%)")

    # 4. Feature/Signal Relationships
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("----------------------------")

    # Relationship between Duration and Number of Labels
    # Does a longer clip imply more sound events?
    corr_dur_labels = audio_df["duration"].corr(audio_df["num_labels"])
    print(f"Correlation (Duration vs. Label Count): {corr_dur_labels:.4f}")

    # Relationship between Class and Duration
    # We associate the duration of a clip with each of its labels
    # Create a flat list of (label, duration)
    label_durations = []
    # We need to map back to the original df order or re-iterate
    # Since audio_df aligns with df (minus failures), we can assume index alignment if no failures
    # To be safe, let's re-zip with the original dataframe assuming 1-to-1 success

    # Add duration to main df for easier analysis
    df["duration"] = audio_df["duration"]

    # Explode labels to analyze duration per class
    df["label_list"] = df["labels"].apply(lambda x: x.split(","))
    exploded_df = df.explode("label_list")

    # Calculate mean duration per class
    class_durations = exploded_df.groupby("label_list")["duration"].mean()

    print("\nClasses with Longest Average Duration:")
    print(
        class_durations.sort_values(ascending=False)
        .head(5)
        .to_string(float_format="%.4f")
    )

    print("\nClasses with Shortest Average Duration:")
    print(
        class_durations.sort_values(ascending=True)
        .head(5)
        .to_string(float_format="%.4f")
    )

    # Check for potential "Short Audio" bias
    short_clips = df[df["duration"] < 1.0]
    if not short_clips.empty:
        print(f"\nShort Clips (< 1.0s): {len(short_clips)} samples")
        short_labels = [lbl for sublist in short_clips["label_list"] for lbl in sublist]
        common_short = Counter(short_labels).most_common(3)
        print(f"  Most common labels in short clips: {common_short}")


if __name__ == "__main__":
    main()
