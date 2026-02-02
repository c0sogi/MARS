import os
import pandas as pd
import numpy as np
import soundfile as sf
import warnings
import random

# Configuration
INPUT_DIR = "./input"
METADATA_FILE = "./metadata/train.csv"
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def print_section(title):
    print(f"\n{title.upper()}")
    print("-" * len(title))


def get_modality(df):
    # Simple heuristic based on file extensions in the training set
    extensions = (
        df["file_path"].apply(lambda x: os.path.splitext(x)[1].lower()).unique()
    )
    if any(ext in [".jpg", ".jpeg", ".png", ".bmp"] for ext in extensions):
        return "Image"
    elif any(ext in [".wav", ".aif", ".aiff", ".mp3", ".flac"] for ext in extensions):
        return "Audio"
    elif any(ext in [".txt"] for ext in extensions):
        return "Text"
    else:
        return "Tabular"


def analyze_target(df):
    print_section("Target Variable Analysis")

    target_col = "label"
    counts = df[target_col].value_counts()
    ratios = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: '{target_col}'")
    print(f"Total Samples: {len(df)}")
    print("\nClass Distribution:")
    for label, count in counts.items():
        ratio = ratios[label]
        print(f"Class {label}: {count} samples ({ratio:.4f})")

    # Check for imbalance
    majority_class_ratio = ratios.max()
    if majority_class_ratio > 0.6:
        print(
            f"\nImbalance Detected: Majority class constitutes {majority_class_ratio:.4f} of the data."
        )
    else:
        print("\nClass Balance: Data is relatively balanced.")


def analyze_audio_data(df):
    print_section("Input Data Analysis (Audio)")

    # Feature storage
    durations = []
    sample_rates = []
    channels_list = []
    subtypes = []  # Proxy for bit depth
    rms_values = []
    peak_values = []
    labels = []

    print(f"Processing {len(df)} audio files to extract metadata statistics...")

    # Iterate through files to extract metadata and basic signal stats
    # Using a limit if dataset is massive, but 20k small files is feasible in < 10 mins
    valid_count = 0

    for idx, row in df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        if not os.path.exists(full_path):
            continue

        try:
            # sf.info is fast for metadata
            info = sf.info(full_path)

            # Read data for signal stats (RMS, Peak)
            # Given small file size (8kB), reading is fast
            data, sr = sf.read(full_path)

            dur = info.duration
            ch = info.channels
            st = info.subtype

            # Calculate signal stats
            if ch > 1:
                # Average channels for mono stats
                sig = np.mean(data, axis=1)
            else:
                sig = data

            rms = np.sqrt(np.mean(sig**2))
            peak = np.max(np.abs(sig))

            durations.append(dur)
            sample_rates.append(sr)
            channels_list.append(ch)
            subtypes.append(st)
            rms_values.append(rms)
            peak_values.append(peak)
            labels.append(row["label"])

            valid_count += 1

        except Exception as e:
            continue

    # Convert to arrays for analysis
    durations = np.array(durations)
    rms_values = np.array(rms_values)
    peak_values = np.array(peak_values)

    # 1. Signal Duration
    print("\n[Signal Duration (seconds)]")
    print(f"Mean: {np.mean(durations):.4f}")
    print(f"Std : {np.std(durations):.4f}")
    print(f"Min : {np.min(durations):.4f}")
    print(f"Max : {np.max(durations):.4f}")

    # 2. Sampling Rates
    print("\n[Sampling Rates]")
    unique_sr, counts_sr = np.unique(sample_rates, return_counts=True)
    for sr, count in zip(unique_sr, counts_sr):
        print(f"{sr} Hz: {count} samples ({count/valid_count:.4f})")

    # 3. Bit Depths / Subtypes
    print("\n[Bit Depths / Subtypes]")
    unique_st, counts_st = np.unique(subtypes, return_counts=True)
    for st, count in zip(unique_st, counts_st):
        print(f"{st}: {count} samples ({count/valid_count:.4f})")

    # 4. Channels
    print("\n[Channels]")
    unique_ch, counts_ch = np.unique(channels_list, return_counts=True)
    for ch, count in zip(unique_ch, counts_ch):
        label_str = "Mono" if ch == 1 else "Stereo" if ch == 2 else f"{ch}-Channel"
        print(f"{label_str} ({ch}): {count} samples ({count/valid_count:.4f})")

    return pd.DataFrame(
        {"duration": durations, "rms": rms_values, "peak": peak_values, "label": labels}
    )


def analyze_relationships(df_features):
    print_section("Feature/Signal Relationships")

    # We analyze the relationship between extracted meta-features and the target
    features = ["duration", "rms", "peak"]
    target = "label"

    print("Correlation with Target (Point-Biserial):")
    print("(Positive values indicate feature is higher for Whale Calls (1))")

    for feat in features:
        # Calculate Point-Biserial Correlation
        # r_pb = (mean1 - mean0) / s_n * sqrt(n1*n0 / n^2)
        # Using numpy corrcoef which is equivalent for binary target
        corr = np.corrcoef(df_features[feat], df_features[target])[0, 1]
        print(f"{feat.capitalize()}: {corr:.4f}")

    print("\nFeature Statistics by Class:")
    for feat in features:
        mean_0 = df_features[df_features[target] == 0][feat].mean()
        mean_1 = df_features[df_features[target] == 1][feat].mean()
        print(
            f"{feat.capitalize()}: Class 0 Mean = {mean_0:.4f}, Class 1 Mean = {mean_1:.4f}"
        )


def main():
    # Load Data
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)

    # Determine Modality
    modality = get_modality(df)

    # 1. Target Analysis
    analyze_target(df)

    # 2. Input Data Analysis & 3. Relationships
    if modality == "Audio":
        df_features = analyze_audio_data(df)
        analyze_relationships(df_features)
    else:
        print(f"\nModality '{modality}' logic not implemented in this specific script.")


if __name__ == "__main__":
    main()
