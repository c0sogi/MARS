import os
import pandas as pd
import numpy as np
import soundfile as sf
import warnings
import random

# 1. Setup and Configuration
# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    input_dir = "./input"
    metadata_path = "./metadata/train.csv"

    # Check if metadata exists
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    # Load Training Data
    df = pd.read_csv(metadata_path)

    # Parse Labels
    # Labels are space-separated strings of integers, e.g., "0 4"
    # We need to convert this to a binary matrix for 19 species
    num_classes = 19

    def parse_labels(label_str):
        if pd.isna(label_str) or label_str == "?":
            return []
        try:
            return [int(x) for x in str(label_str).split()]
        except ValueError:
            return []

    df["label_list"] = df["labels"].apply(parse_labels)

    # Create Binary Label Matrix
    label_matrix = np.zeros((len(df), num_classes), dtype=int)
    for idx, labels in enumerate(df["label_list"]):
        for lbl in labels:
            if 0 <= lbl < num_classes:
                label_matrix[idx, lbl] = 1

    # Add label count as a meta-feature
    df["num_labels"] = df["label_list"].apply(len)

    print("SECTION 1: DATA INTEGRITY")
    print(f"Analysis performed strictly on Training Set: {metadata_path}")
    print(f"Number of training samples: {len(df)}")
    print("-" * 30)

    # 2. Target Variable Analysis
    print("SECTION 2: TARGET VARIABLE ANALYSIS")

    # Class Counts
    class_counts = label_matrix.sum(axis=0)
    total_samples = len(df)

    print(f"Target Type: Multi-label Classification (19 Species)")
    print(f"Class Distribution (Counts per Species 0-18):")
    print(f"{class_counts}")

    # Imbalance
    min_count = class_counts.min()
    max_count = class_counts.max()
    mean_count = class_counts.mean()

    # Avoid division by zero
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    print(f"Minimum Class Count: {min_count}")
    print(f"Maximum Class Count: {max_count}")
    print(f"Mean Class Count: {mean_count:.4f}")
    print(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.4f}")

    # Label Cardinality (Average number of labels per sample)
    label_cardinality = df["num_labels"].mean()
    print(f"Label Cardinality (Avg labels/sample): {label_cardinality:.4f}")

    # Samples with no labels
    no_label_count = (df["num_labels"] == 0).sum()
    print(
        f"Samples with no labels: {no_label_count} ({no_label_count/total_samples*100:.2f}%)"
    )
    print("-" * 30)

    # 3. Input Data Analysis (Audio)
    print("SECTION 3: INPUT DATA ANALYSIS (AUDIO)")

    audio_stats = []

    # We need to read the audio files.
    # The file_path in metadata is relative to input_dir.
    # Example: essential_data/src_wavs/filename.wav

    pixel_means = []
    pixel_stds = []

    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        try:
            # Use soundfile to get info and data
            info = sf.info(full_path)
            data, samplerate = sf.read(full_path)

            # Handle multi-channel if necessary (though dataset desc says mono)
            if len(data.shape) > 1:
                channels = data.shape[1]
                # Flatten for stats
                flat_data = data.flatten()
            else:
                channels = 1
                flat_data = data

            duration = info.duration
            subtype = info.subtype  # Bit depth info usually

            # Signal Stats
            sig_mean = np.mean(flat_data)
            sig_std = np.std(flat_data)
            sig_min = np.min(flat_data)
            sig_max = np.max(flat_data)
            # Energy (Mean Squared Amplitude)
            sig_energy = np.mean(flat_data**2)

            audio_stats.append(
                {
                    "duration": duration,
                    "samplerate": samplerate,
                    "channels": channels,
                    "subtype": subtype,
                    "mean": sig_mean,
                    "std": sig_std,
                    "min": sig_min,
                    "max": sig_max,
                    "energy": sig_energy,
                }
            )

            pixel_means.append(sig_mean)
            pixel_stds.append(sig_std)

        except Exception as e:
            # In case of missing file or read error, though verification passed
            continue

    stats_df = pd.DataFrame(audio_stats)

    if not stats_df.empty:
        # Signal Analysis
        print("Signal Properties:")
        print(f"  Duration Mean: {stats_df['duration'].mean():.4f} s")
        print(f"  Duration Std:  {stats_df['duration'].std():.4f} s")
        print(f"  Min Duration:  {stats_df['duration'].min():.4f} s")
        print(f"  Max Duration:  {stats_df['duration'].max():.4f} s")

        print(f"  Sampling Rates: {stats_df['samplerate'].unique()}")
        print(f"  Bit Depths (Subtypes): {stats_df['subtype'].unique()}")

        # Channels
        channel_counts = stats_df["channels"].value_counts()
        print(f"  Channel Distribution: {channel_counts.to_dict()}")

        # Signal Statistics (Global)
        # Note: Averaging means and stds of files is an approximation of global stats
        global_mean = np.mean(pixel_means)
        global_std = np.mean(pixel_stds)  # Average of STDs is a rough proxy

        print("Signal Statistics (Amplitude):")
        print(f"  Global Mean: {global_mean:.6f}")
        print(f"  Global Std Dev: {global_std:.6f}")
        print(f"  Min Amplitude Observed: {stats_df['min'].min():.4f}")
        print(f"  Max Amplitude Observed: {stats_df['max'].max():.4f}")
    else:
        print("No audio data could be processed.")
    print("-" * 30)

    # 4. Feature/Signal Relationships
    print("SECTION 4: FEATURE/SIGNAL RELATIONSHIPS")

    if not stats_df.empty:
        # Combine stats with labels
        # Assuming index alignment is preserved (it should be as we iterated df)
        # If any files failed, we need to align carefully.
        # Since verification passed, we assume 1-to-1.

        analysis_df = pd.concat(
            [df.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1
        )

        # Unstructured (Meta-Feature) Relationships
        # 1. Correlation between Signal Energy and Number of Labels
        # Hypothesis: More birds -> Higher energy or complexity?
        corr_energy_labels = analysis_df["energy"].corr(analysis_df["num_labels"])
        print(f"Correlation (Signal Energy vs. Label Count): {corr_energy_labels:.4f}")

        # 2. Correlation between Signal Std (Dynamic Range proxy) and Label Count
        corr_std_labels = analysis_df["std"].corr(analysis_df["num_labels"])
        print(f"Correlation (Signal Std vs. Label Count): {corr_std_labels:.4f}")

        # 3. Check if specific classes have distinct energy profiles
        # Calculate point-biserial correlation for the most common class
        most_common_class_idx = np.argmax(class_counts)
        # Extract binary vector for this class
        y_common = label_matrix[:, most_common_class_idx]

        # Filter analysis_df to match label_matrix length (should be same)
        if len(analysis_df) == len(y_common):
            corr_class_energy = np.corrcoef(analysis_df["energy"], y_common)[0, 1]
            print(
                f"Correlation (Signal Energy vs. Presence of Most Common Class {most_common_class_idx}): {corr_class_energy:.4f}"
            )

        # 4. Check for correlation between Duration and Labels (likely 0 variance if all 10s)
        if analysis_df["duration"].std() > 0:
            corr_dur_labels = analysis_df["duration"].corr(analysis_df["num_labels"])
            print(f"Correlation (Duration vs. Label Count): {corr_dur_labels:.4f}")
        else:
            print("Correlation (Duration vs. Label Count): N/A (Duration is constant)")

    else:
        print("Insufficient data for relationship analysis.")


if __name__ == "__main__":
    main()
