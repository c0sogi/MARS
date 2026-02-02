import os
import pandas as pd
import numpy as np
import soundfile as sf
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Constants
INPUT_DIR = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def analyze_target(df):
    print("SECTION 1: TARGET VARIABLE ANALYSIS")
    target_col = "label"

    counts = df[target_col].value_counts()
    proportions = df[target_col].value_counts(normalize=True)

    print(f"Target Variable: '{target_col}'")
    print(f"Total Samples: {len(df)}")
    print("Class Distribution:")
    for label, count in counts.items():
        prop = proportions[label]
        print(f"  Class {label}: {count} samples ({prop:.4f})")

    # Imbalance check
    ratio = counts.min() / counts.max()
    print(f"Class Imbalance Ratio (Minority/Majority): {ratio:.4f}")
    print("-" * 30)


def extract_audio_features(df):
    results = []

    # Iterate through files
    # Note: Using a loop here as we need to open files to get duration/channels/signal stats
    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)
        label = row["label"]

        try:
            # Read audio file
            # sf.read returns (data, samplerate)
            # data is numpy array (frames, channels) or (frames,)
            data, samplerate = sf.read(full_path)

            # Determine dimensions
            if data.ndim == 1:
                n_channels = 1
                n_frames = data.shape[0]
                flat_data = data
            else:
                n_channels = data.shape[1]
                n_frames = data.shape[0]
                flat_data = data.flatten()

            duration = n_frames / samplerate

            # Signal stats
            # Mean of absolute amplitude (volume proxy)
            mean_amp = np.mean(np.abs(flat_data))
            # Std dev of amplitude (dynamic range proxy)
            std_amp = np.std(flat_data)

            results.append(
                {
                    "duration": duration,
                    "sample_rate": samplerate,
                    "channels": n_channels,
                    "mean_amp": mean_amp,
                    "std_amp": std_amp,
                    "label": label,
                }
            )

        except Exception as e:
            # Skip files that cannot be read
            continue

    return pd.DataFrame(results)


def analyze_audio_data(df_features):
    print("SECTION 2: INPUT DATA ANALYSIS (AUDIO)")

    # 1. Signal Duration
    durations = df_features["duration"]
    print("Duration Statistics (seconds):")
    print(f"  Mean: {durations.mean():.4f}")
    print(f"  Std : {durations.std():.4f}")
    print(f"  Min : {durations.min():.4f}")
    print(f"  Max : {durations.max():.4f}")

    # 2. Sampling Rates
    print("\nSampling Rate Distribution:")
    sr_counts = df_features["sample_rate"].value_counts(normalize=True)
    for sr, prop in sr_counts.items():
        print(f"  {sr} Hz: {prop:.4f}")

    # 3. Channels
    print("\nChannel Count Distribution:")
    ch_counts = df_features["channels"].value_counts(normalize=True)
    for ch, prop in ch_counts.items():
        print(f"  {ch} channel(s): {prop:.4f}")

    # 4. Signal Statistics
    print("\nGlobal Signal Statistics:")
    print(f"  Mean Absolute Amplitude: {df_features['mean_amp'].mean():.4f}")
    print(f"  Mean Signal Std Dev: {df_features['std_amp'].mean():.4f}")
    print("-" * 30)


def analyze_relationships(df_features):
    print("SECTION 3: FEATURE/SIGNAL RELATIONSHIPS")

    # Correlation with Target
    # Since target is binary (0/1), Pearson correlation is equivalent to Point-Biserial
    correlations = df_features[
        ["duration", "sample_rate", "channels", "mean_amp", "std_amp", "label"]
    ].corr()["label"]

    print("Correlation with Target (Label):")
    for feat in ["duration", "sample_rate", "channels", "mean_amp", "std_amp"]:
        if feat in correlations:
            print(f"  {feat}: {correlations[feat]:.4f}")

    # Compare means by class
    print("\nFeature Means by Class:")
    grouped = df_features.groupby("label")[["duration", "mean_amp", "std_amp"]].mean()

    for label in grouped.index:
        print(f"  Class {label}:")
        print(f"    Avg Duration: {grouped.loc[label, 'duration']:.4f}s")
        print(f"    Avg Mean Amp: {grouped.loc[label, 'mean_amp']:.4f}")
        print(f"    Avg Std Amp : {grouped.loc[label, 'std_amp']:.4f}")

    # Check for redundancy (collinearity) between meta-features
    print("\nMeta-Feature Redundancy (Correlation > 0.90):")
    feature_subset = df_features[["duration", "mean_amp", "std_amp"]]
    corr_matrix = feature_subset.corr().abs()

    found_redundancy = False
    # Iterate over lower triangle
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > 0.90:
                print(
                    f"  High correlation ({corr_matrix.iloc[i, j]:.4f}) between {corr_matrix.columns[i]} and {corr_matrix.columns[j]}"
                )
                found_redundancy = True

    if not found_redundancy:
        print("  No highly collinear meta-feature pairs found.")

    print("-" * 30)


def main():
    set_seed(SEED)

    # Load Metadata
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df_meta = pd.read_csv(METADATA_PATH)

    # 1. Target Analysis
    analyze_target(df_meta)

    # 2. Extract Audio Features
    # We perform extraction once and use the resulting dataframe for subsequent steps
    df_features = extract_audio_features(df_meta)

    if df_features.empty:
        print("Error: No audio features extracted. Check file paths.")
        return

    # 3. Input Data Analysis
    analyze_audio_data(df_features)

    # 4. Feature Relationships
    analyze_relationships(df_features)


if __name__ == "__main__":
    main()
