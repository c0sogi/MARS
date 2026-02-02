import os
import pandas as pd
import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
INPUT_ROOT = "./input"
METADATA_PATH = "./metadata/train.csv"
SEED = 42


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def analyze_target(df):
    """Analyze the distribution of the target variable."""
    print("TARGET VARIABLE ANALYSIS")

    if "label" not in df.columns:
        print("Error: 'label' column not found in metadata.")
        return

    counts = df["label"].value_counts()
    props = df["label"].value_counts(normalize=True)

    print(f"Class Distribution:\n{counts.to_string()}")
    print(f"Class Ratios:\n{props.to_string()}")

    # Imbalance check
    if len(counts) > 0:
        ratio = counts.min() / counts.max()
        print(f"Minority/Majority Ratio: {ratio:.4f}")
        if ratio < 0.2:
            print("Observation: The dataset is significantly imbalanced.")
        else:
            print("Observation: The dataset is relatively balanced.")
    print("-" * 30)


def analyze_audio_metadata(df):
    """Analyze technical properties of the audio files."""
    print("INPUT DATA ANALYSIS (AUDIO)")

    durations = []
    sample_rates = []
    channels = []
    subtypes = []

    # We analyze all files as header reading is fast
    print(f"Scanning metadata for {len(df)} audio files...")

    for _, row in df.iterrows():
        full_path = os.path.join(INPUT_ROOT, row["filepath"])
        if not os.path.exists(full_path):
            continue

        try:
            # soundfile.info is efficient as it only reads headers
            info = sf.info(full_path)
            durations.append(info.duration)
            sample_rates.append(info.samplerate)
            channels.append(info.channels)
            subtypes.append(info.subtype)
        except Exception:
            continue

    if not durations:
        print("No audio files could be analyzed.")
        return

    # Signal Analysis
    durations = np.array(durations)
    print(
        f"Duration (s) - Mean: {np.mean(durations):.4f}, Std: {np.std(durations):.4f}, Min: {np.min(durations):.4f}, Max: {np.max(durations):.4f}"
    )

    # Sampling Rates
    sr_counts = pd.Series(sample_rates).value_counts()
    print(f"Sampling Rates Distribution:\n{sr_counts.to_string()}")

    # Bit Depths / Subtypes
    subtype_counts = pd.Series(subtypes).value_counts()
    print(f"Bit Depth/Subtype Distribution:\n{subtype_counts.to_string()}")

    # Channels Analysis
    ch_counts = pd.Series(channels).value_counts()
    print(f"Channel Counts Distribution:\n{ch_counts.to_string()}")
    if len(ch_counts) > 1:
        print("Warning: Inconsistent channel counts detected (Mono vs Stereo).")
    else:
        print("Consistent channel counts.")
    print("-" * 30)


def extract_features_for_sample(filepath):
    """Extract basic spectral and temporal features from an audio file."""
    try:
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(filepath)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze()

        # 1. RMS Energy
        rms = torch.sqrt(torch.mean(waveform**2)).item()

        # 2. Max Amplitude
        max_amp = torch.max(torch.abs(waveform)).item()

        # 3. Zero Crossing Rate
        if len(waveform) > 1:
            zcr = ((waveform[:-1] * waveform[1:]) < 0).float().mean().item()
        else:
            zcr = 0.0

        # 4. Spectral Centroid Proxy (using FFT)
        fft = torch.fft.rfft(waveform)
        magnitude = fft.abs()
        freqs = torch.fft.rfftfreq(len(waveform), 1 / sample_rate)

        if magnitude.sum() > 0:
            centroid = torch.sum(freqs * magnitude) / torch.sum(magnitude)
            centroid = centroid.item()
        else:
            centroid = 0.0

        # 5. Spectral Flatness Proxy
        mag_sq = magnitude**2
        if mag_sq.mean() > 0:
            gmean = torch.exp(torch.mean(torch.log(mag_sq + 1e-10))).item()
            amean = torch.mean(mag_sq).item()
            flatness = gmean / (amean + 1e-10)
        else:
            flatness = 0.0

        return {
            "rms": rms,
            "max_amp": max_amp,
            "zcr": zcr,
            "centroid": centroid,
            "flatness": flatness,
            "duration": len(waveform) / sample_rate,
        }
    except Exception:
        return None


def analyze_relationships(df):
    """Analyze relationships between features and the target variable."""
    print("FEATURE/SIGNAL RELATIONSHIPS")

    # Use a stratified subset for feature extraction to ensure efficiency
    sample_n = min(2000, len(df))
    try:
        df_sample, _ = train_test_split(
            df, train_size=sample_n, stratify=df["label"], random_state=SEED
        )
    except ValueError:
        df_sample = df.sample(n=sample_n, random_state=SEED)

    print(
        f"Extracting features for {len(df_sample)} samples to analyze relationships..."
    )

    data_list = []
    for _, row in df_sample.iterrows():
        full_path = os.path.join(INPUT_ROOT, row["filepath"])
        feats = extract_features_for_sample(full_path)
        if feats:
            feats["label"] = row["label"]
            data_list.append(feats)

    df_feats = pd.DataFrame(data_list)

    if df_feats.empty:
        print("No features extracted.")
        return

    # 1. Unstructured (Meta-Feature) Relationships
    print("\nMeta-Feature Analysis (Duration vs Target):")
    # Compare duration stats by label
    duration_stats = df_feats.groupby("label")["duration"].describe()[
        ["mean", "std", "min", "max"]
    ]
    print(duration_stats.to_string())

    # 2. Structured Relationships (on extracted features)
    print("\nStructured Feature Analysis (Extracted Features):")
    feature_cols = [c for c in df_feats.columns if c != "label"]

    # Correlation
    corr_matrix = df_feats[feature_cols].corr()

    # Redundancy Check
    print("Redundant Pairs (Correlation > 0.90):")
    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            c1 = feature_cols[i]
            c2 = feature_cols[j]
            val = corr_matrix.loc[c1, c2]
            if abs(val) > 0.90:
                pairs.append(f"{c1} & {c2}: {val:.4f}")

    if pairs:
        for p in pairs:
            print(p)
    else:
        print("None detected.")

    # Feature Importance (Random Forest)
    print("\nTop 5 Features (Random Forest Importance):")
    X = df_feats[feature_cols]
    y = df_feats["label"]

    rf = RandomForestClassifier(
        n_estimators=100, random_state=SEED, n_jobs=-1, max_depth=10
    )
    rf.fit(X, y)

    imps = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    print(imps.head(5).to_string())


def main():
    set_seed(SEED)

    if not os.path.exists(METADATA_PATH):
        print(f"Metadata file not found at {METADATA_PATH}")
        return

    df_train = pd.read_csv(METADATA_PATH)

    # Execute Analysis Sections
    analyze_target(df_train)
    analyze_audio_metadata(df_train)
    analyze_relationships(df_train)


if __name__ == "__main__":
    main()
