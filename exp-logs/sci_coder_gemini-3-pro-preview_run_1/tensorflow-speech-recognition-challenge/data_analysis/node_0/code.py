import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def analyze_audio_data():
    # Constants
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"
    SAMPLE_SIZE = 2000  # Number of files to sample for detailed signal analysis
    SEED = 42

    set_seed(SEED)

    print("DATA INTEGRITY CHECK")
    print("-" * 20)
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    # Load Metadata
    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded training metadata with {len(df)} records.")

    # ---------------------------------------------------------
    # 2. Target Variable Analysis
    # ---------------------------------------------------------
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 20)

    # Distribution
    label_counts = df["label"].value_counts()
    print("Label Distribution:")
    print(label_counts.to_string())

    # Imbalance
    min_class = label_counts.min()
    max_class = label_counts.max()
    balance_ratio = min_class / max_class
    print(f"\nClass Balance Ratio (Min/Max): {balance_ratio:.4f}")

    most_freq = label_counts.idxmax()
    least_freq = label_counts.idxmin()
    print(f"Most Frequent Class: {most_freq} ({max_class})")
    print(f"Least Frequent Class: {least_freq} ({min_class})")

    # ---------------------------------------------------------
    # 3. Input Data Analysis (Audio)
    # ---------------------------------------------------------
    print("\nINPUT DATA ANALYSIS (AUDIO)")
    print("-" * 20)

    # Stratified Sampling for efficient analysis
    # We filter out files that don't exist just in case, though metadata should be clean
    # We also handle the case where a class has fewer samples than n_splits in stratified sampling
    # by just taking a simple random sample if stratified fails or doing manual per-group sampling.

    # Simple strategy: Sample proportional to distribution, minimum 1 per class if possible.
    if len(df) > SAMPLE_SIZE:
        df_sample = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(frac=SAMPLE_SIZE / len(df), random_state=SEED)
        )
        # If sampling resulted in too few (due to rounding), fill up randomly
        if len(df_sample) < SAMPLE_SIZE:
            remaining = df.drop(df_sample.index)
            n_needed = SAMPLE_SIZE - len(df_sample)
            if not remaining.empty:
                extra = remaining.sample(
                    n=min(n_needed, len(remaining)), random_state=SEED
                )
                df_sample = pd.concat([df_sample, extra])
    else:
        df_sample = df.copy()

    print(
        f"Analyzing a stratified sample of {len(df_sample)} files for signal properties..."
    )

    audio_stats = []

    for idx, row in df_sample.iterrows():
        filepath = os.path.join(INPUT_DIR, row["filepath"])
        label = row["label"]

        try:
            # Metadata using soundfile (fast)
            info = sf.info(filepath)
            duration = info.duration
            sr = info.samplerate
            channels = info.channels
            subtype = info.subtype  # indicative of bit depth

            # Signal features using librosa (slower, but necessary for content analysis)
            # Load with native sr to avoid resampling overhead for now, or fixed for consistency.
            # We use a fixed duration load to speed up if files are long (background noise)
            y, s = librosa.load(filepath, sr=None, duration=5.0)

            # Extract simple features
            rms = np.mean(librosa.feature.rms(y=y))
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=s))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

            audio_stats.append(
                {
                    "label": label,
                    "duration": duration,
                    "samplerate": sr,
                    "channels": channels,
                    "subtype": subtype,
                    "rms": rms,
                    "spectral_centroid": spec_cent,
                    "zero_crossing_rate": zcr,
                }
            )

        except Exception as e:
            # Skip corrupted files in analysis
            continue

    df_stats = pd.DataFrame(audio_stats)

    # Signal: Duration
    dur_mean = df_stats["duration"].mean()
    dur_std = df_stats["duration"].std()
    dur_min = df_stats["duration"].min()
    dur_max = df_stats["duration"].max()

    print(f"\nDuration (seconds):")
    print(f"  Mean: {dur_mean:.4f}")
    print(f"  Std : {dur_std:.4f}")
    print(f"  Min : {dur_min:.4f}")
    print(f"  Max : {dur_max:.4f}")

    # Signal: Sampling Rates
    print(f"\nSampling Rates Distribution:")
    print(df_stats["samplerate"].value_counts().to_string())

    # Signal: Bit Depths (Subtypes)
    print(f"\nBit Depth/Subtype Distribution:")
    print(df_stats["subtype"].value_counts().to_string())

    # Channels
    print(f"\nChannel Counts:")
    print(df_stats["channels"].value_counts().to_string())
    if len(df_stats["channels"].unique()) > 1:
        print("  WARNING: Inconsistent channel counts detected (Mono vs Stereo).")
    else:
        print("  Consistent channel counts detected.")

    # ---------------------------------------------------------
    # 4. Feature/Signal Relationships
    # ---------------------------------------------------------
    print("\nFEATURE/SIGNAL RELATIONSHIPS")
    print("-" * 20)

    # Meta-Feature Relationship: Duration vs Label
    # We check if specific labels tend to be longer/shorter (excluding silence which is known to be long)
    print("Average Duration by Label (Top 5 Longest):")
    dur_by_label = (
        df_stats.groupby("label")["duration"].mean().sort_values(ascending=False)
    )
    print(dur_by_label.head(5).to_string(float_format="{:.4f}".format))

    # Structured Relationship: Feature Importance via Random Forest
    print("\nFeature Importance (Random Forest):")

    # Prepare data for RF
    feature_cols = ["duration", "rms", "spectral_centroid", "zero_crossing_rate"]
    X = df_stats[feature_cols].fillna(0)
    y = df_stats["label"]

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=50, random_state=SEED, n_jobs=-1, max_depth=10
    )
    rf.fit(X, y_enc)

    # Get importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top Features predicting Label:")
    for f in range(len(feature_cols)):
        idx = indices[f]
        print(f"  {f+1}. {feature_cols[idx]}: {importances[idx]:.4f}")

    # Correlation Check (Collinearity among features)
    print("\nFeature Correlation Matrix (Pearson):")
    corr_matrix = X.corr()
    print(corr_matrix.to_string(float_format="{:.4f}".format))

    # Check for high redundancy
    high_corr = []
    cols = X.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) > 0.90:
                high_corr.append((cols[i], cols[j], corr_matrix.iloc[i, j]))

    if high_corr:
        print("\nRedundant Features (Correlation > 0.90):")
        for c1, c2, val in high_corr:
            print(f"  {c1} - {c2}: {val:.4f}")
    else:
        print("\nNo highly collinear features (> 0.90) found among extracted signals.")


if __name__ == "__main__":
    analyze_audio_data()
