import os
import numpy as np
import pandas as pd
import soundfile as sf
import random
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.stats import skew, kurtosis
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
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_targets(df):
    print("==== TARGET VARIABLE ANALYSIS ====")

    # Parse labels
    df["label_list"] = df["labels"].apply(lambda x: x.split(","))

    # Binarize labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["label_list"])
    classes = mlb.classes_

    # 1. Distribution & Imbalance
    label_counts = np.sum(y, axis=0)
    total_samples = len(df)

    # Create a dataframe for label stats
    label_stats = pd.DataFrame(
        {
            "Label": classes,
            "Count": label_counts,
            "Frequency": label_counts / total_samples,
        }
    ).sort_values("Count", ascending=False)

    print(f"Total Unique Classes: {len(classes)}")
    print(f"Total Samples: {total_samples}")

    print("\n-- Class Balance --")
    max_count = label_stats["Count"].max()
    min_count = label_stats["Count"].min()
    mean_count = label_stats["Count"].mean()

    print(
        f"Most Frequent Label: {label_stats.iloc[0]['Label']} ({label_stats.iloc[0]['Count']} samples, {label_stats.iloc[0]['Frequency']:.4f})"
    )
    print(
        f"Least Frequent Label: {label_stats.iloc[-1]['Label']} ({label_stats.iloc[-1]['Count']} samples, {label_stats.iloc[-1]['Frequency']:.4f})"
    )
    print(f"Mean Samples per Label: {mean_count:.4f}")
    print(f"Imbalance Ratio (Max/Min): {max_count/min_count:.4f}")

    # Rare labels (< 1%)
    rare_labels = label_stats[label_stats["Frequency"] < 0.01]
    print(f"Count of Rare Labels (< 1% freq): {len(rare_labels)}")
    if len(rare_labels) > 0:
        print(f"Top 3 Rare Labels: {', '.join(rare_labels['Label'].head(3).tolist())}")

    # 2. Label Cardinality (Labels per clip)
    labels_per_clip = np.sum(y, axis=1)
    print("\n-- Label Cardinality --")
    print(f"Mean Labels per Clip: {np.mean(labels_per_clip):.4f}")
    print(f"Max Labels per Clip: {np.max(labels_per_clip)}")
    print(f"Min Labels per Clip: {np.min(labels_per_clip)}")

    # Return data for relationship analysis
    return y, classes, labels_per_clip


def analyze_audio_metadata(df):
    print("\n==== INPUT DATA ANALYSIS (AUDIO) ====")

    durations = []
    sample_rates = []
    channels = []
    subtypes = []  # Bit depth proxy

    # We iterate through files to get metadata
    # Using soundfile.info is fast as it reads headers only

    valid_indices = []

    for idx, row in df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["file_path"])
        try:
            info = sf.info(full_path)
            durations.append(info.duration)
            sample_rates.append(info.samplerate)
            channels.append(info.channels)
            subtypes.append(info.subtype)
            valid_indices.append(idx)
        except Exception as e:
            # In case a file is unreadable (though metadata generation should have caught this)
            continue

    durations = np.array(durations)
    sample_rates = np.array(sample_rates)
    channels = np.array(channels)

    # 1. Signal Duration
    print("-- Duration (Seconds) --")
    print(f"Mean: {np.mean(durations):.4f}")
    print(f"Std:  {np.std(durations):.4f}")
    print(f"Min:  {np.min(durations):.4f}")
    print(f"Max:  {np.max(durations):.4f}")

    # 2. Sampling Rates
    print("\n-- Sampling Rates --")
    unique_sr, counts_sr = np.unique(sample_rates, return_counts=True)
    for sr, count in zip(unique_sr, counts_sr):
        print(f"{sr} Hz: {count} files ({count/len(sample_rates):.4%})")

    # 3. Channels
    print("\n-- Channels --")
    unique_ch, counts_ch = np.unique(channels, return_counts=True)
    for ch, count in zip(unique_ch, counts_ch):
        label = "Mono" if ch == 1 else "Stereo" if ch == 2 else f"{ch}-Channel"
        print(f"{label} ({ch}): {count} files ({count/len(channels):.4%})")

    # 4. Bit Depths (Subtypes)
    print("\n-- Bit Depths / Subtypes --")
    unique_sub, counts_sub = np.unique(subtypes, return_counts=True)
    for sub, count in zip(unique_sub, counts_sub):
        print(f"{sub}: {count} files")

    return durations, valid_indices


def analyze_relationships(y, classes, labels_per_clip, durations, valid_indices):
    print("\n==== FEATURE/SIGNAL RELATIONSHIPS ====")

    # Filter y and labels_per_clip to match valid audio files
    y_valid = y[valid_indices]
    labels_per_clip_valid = labels_per_clip[valid_indices]

    # 1. Meta-Feature Relationships (Duration vs Targets)
    print("-- Metadata vs Target --")

    # Correlation between Duration and Number of Labels
    corr_cardinality = np.corrcoef(durations, labels_per_clip_valid)[0, 1]
    print(f"Correlation (Duration vs Label Count): {corr_cardinality:.4f}")

    # Correlation between Duration and Specific Classes
    # Do longer files correlate with specific classes?
    # We calculate Point-Biserial correlation (Continuous vs Binary)

    correlations = []
    for i, label in enumerate(classes):
        # Avoid division by zero if variance is 0 (though unlikely given target analysis)
        if np.std(y_valid[:, i]) > 0:
            corr = np.corrcoef(durations, y_valid[:, i])[0, 1]
            correlations.append((label, corr))
        else:
            correlations.append((label, 0.0))

    correlations.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 3 Classes Positively Correlated with Duration:")
    for label, corr in correlations[:3]:
        print(f"  {label}: {corr:.4f}")

    print("\nTop 3 Classes Negatively Correlated with Duration:")
    for label, corr in correlations[-3:]:
        print(f"  {label}: {corr:.4f}")


def main():
    set_seed(SEED)

    # Load Data
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    # Run Analysis
    y, classes, labels_per_clip = analyze_targets(df)
    durations, valid_indices = analyze_audio_metadata(df)
    analyze_relationships(y, classes, labels_per_clip, durations, valid_indices)


if __name__ == "__main__":
    main()
