import os
import pandas as pd
import numpy as np
import soundfile as sf
import random

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"


def run_eda():
    # 1. Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Identify label columns
    # Columns that are not metadata
    meta_cols = ["fname", "labels", "filepath"]
    label_cols = [c for c in df.columns if c not in meta_cols]

    print("=== TARGET VARIABLE ANALYSIS ===")

    # Class Balance Analysis
    label_counts = df[label_cols].sum().sort_values(ascending=False)
    total_samples = len(df)

    print(f"Total Samples: {total_samples}")
    print(f"Total Classes: {len(label_cols)}")

    # Calculate ratios
    label_ratios = label_counts / total_samples

    print("\n--- Class Balance ---")
    print(
        f"Most Frequent Class: {label_counts.index[0]} ({label_counts.iloc[0]} samples, {label_ratios.iloc[0]:.4f})"
    )
    print(
        f"Least Frequent Class: {label_counts.index[-1]} ({label_counts.iloc[-1]} samples, {label_ratios.iloc[-1]:.4f})"
    )
    print(f"Mean Class Frequency: {label_counts.mean():.4f}")
    print(f"Std Dev Class Frequency: {label_counts.std():.4f}")

    # Multi-label Analysis (Cardinality)
    df["label_count"] = df[label_cols].sum(axis=1)
    print("\n--- Multi-label Cardinality ---")
    print(f"Mean Labels per Sample: {df['label_count'].mean():.4f}")
    print(f"Max Labels per Sample: {df['label_count'].max()}")
    print(f"Min Labels per Sample: {df['label_count'].min()}")

    # Distribution of label counts
    cardinality_dist = df["label_count"].value_counts().sort_index()
    print("Label Count Distribution (Labels per Clip -> Count):")
    for k, v in cardinality_dist.items():
        print(f"  {k} labels: {v} samples ({v/total_samples:.4f})")

    print("\n=== AUDIO DATA ANALYSIS ===")

    # Extract Audio Metadata
    # We will read headers of files to get duration, samplerate, channels
    durations = []
    sample_rates = []
    channels = []

    # Use a counter to track missing files if any (though metadata should be clean)
    missing_files = 0

    # To save time if dataset is massive, we could sample, but 23k is manageable for header reading.
    # We will process all to be robust.

    print("Extracting audio metadata from files...")
    for idx, row in df.iterrows():
        # Construct full path
        full_path = os.path.join(INPUT_DIR, row["filepath"])

        try:
            # sf.info reads only the header
            info = sf.info(full_path)
            durations.append(info.duration)
            sample_rates.append(info.samplerate)
            channels.append(info.channels)
        except Exception as e:
            missing_files += 1
            durations.append(np.nan)
            sample_rates.append(np.nan)
            channels.append(np.nan)

    # Convert to numpy arrays for analysis
    durations = np.array(durations, dtype=float)
    sample_rates = np.array(sample_rates, dtype=float)
    channels = np.array(channels, dtype=float)

    # Add to dataframe for relationship analysis
    df["duration"] = durations
    df["sample_rate"] = sample_rates
    df["n_channels"] = channels

    # Filter out NaNs for stats
    valid_durations = durations[~np.isnan(durations)]
    valid_sr = sample_rates[~np.isnan(sample_rates)]
    valid_ch = channels[~np.isnan(channels)]

    if missing_files > 0:
        print(f"Warning: {missing_files} audio files could not be read.")

    # Duration Stats
    print("\n--- Duration (seconds) ---")
    print(f"Mean: {np.mean(valid_durations):.4f}")
    print(f"Std:  {np.std(valid_durations):.4f}")
    print(f"Min:  {np.min(valid_durations):.4f}")
    print(f"Max:  {np.max(valid_durations):.4f}")

    # Sample Rate Stats
    print("\n--- Sample Rates ---")
    unique_sr, counts_sr = np.unique(valid_sr, return_counts=True)
    # Sort by count desc
    sorted_indices = np.argsort(-counts_sr)
    for i in sorted_indices:
        print(
            f"  {int(unique_sr[i])} Hz: {counts_sr[i]} files ({(counts_sr[i]/len(valid_sr)):.4f})"
        )

    # Channel Stats
    print("\n--- Channels ---")
    unique_ch, counts_ch = np.unique(valid_ch, return_counts=True)
    sorted_indices_ch = np.argsort(-counts_ch)
    for i in sorted_indices_ch:
        ch_type = (
            "Mono"
            if unique_ch[i] == 1
            else "Stereo" if unique_ch[i] == 2 else f"{int(unique_ch[i])}-channel"
        )
        print(
            f"  {ch_type} ({int(unique_ch[i])}): {counts_ch[i]} files ({(counts_ch[i]/len(valid_ch)):.4f})"
        )

    print("\n=== FEATURE/SIGNAL RELATIONSHIPS ===")

    # Relationship between Duration and Number of Labels
    # Drop NaNs
    df_clean = df.dropna(subset=["duration", "label_count"])

    corr = df_clean["duration"].corr(df_clean["label_count"])
    print(f"Correlation between Audio Duration and Number of Labels: {corr:.4f}")

    # Analyze if specific classes have distinct duration profiles
    # We calculate the point-biserial correlation proxy:
    # Mean duration of positive samples vs Mean duration of negative samples for each class

    print("\n--- Duration vs Class Presence ---")
    class_duration_diffs = []

    global_mean_duration = df_clean["duration"].mean()

    for label in label_cols:
        pos_mask = df_clean[label] == 1
        if pos_mask.sum() > 0:
            avg_dur_pos = df_clean.loc[pos_mask, "duration"].mean()
            diff = avg_dur_pos - global_mean_duration
            class_duration_diffs.append((label, avg_dur_pos, diff))

    # Sort by absolute difference from global mean
    class_duration_diffs.sort(key=lambda x: x[1], reverse=True)

    print(f"Global Mean Duration: {global_mean_duration:.4f}s")
    print("Classes with Longest Average Duration:")
    for label, avg, diff in class_duration_diffs[:5]:
        print(f"  {label}: {avg:.4f}s (Diff: {diff:+.4f}s)")

    print("\nClasses with Shortest Average Duration:")
    for label, avg, diff in class_duration_diffs[-5:]:
        print(f"  {label}: {avg:.4f}s (Diff: {diff:+.4f}s)")


if __name__ == "__main__":
    run_eda()
