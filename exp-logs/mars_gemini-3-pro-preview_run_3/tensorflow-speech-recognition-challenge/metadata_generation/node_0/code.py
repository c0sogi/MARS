import os
import glob
import pandas as pd
import numpy as np

# Configuration
INPUT_DIR = "./input"
TRAIN_AUDIO_DIR = os.path.join(INPUT_DIR, "train", "audio")
TEST_AUDIO_DIR = os.path.join(INPUT_DIR, "test", "audio")
METADATA_DIR = "./metadata"
RANDOM_STATE = 42

TARGET_LABELS = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}


def get_label_from_folder(folder_name):
    """Maps folder names to target labels, unknown, or silence."""
    if folder_name == "_background_noise_":
        return "silence"
    if folder_name in TARGET_LABELS:
        return folder_name
    return "unknown"


def parse_training_data():
    """Scans training directory and extracts metadata."""
    records = []

    # Get all subdirectories in train/audio
    if not os.path.exists(TRAIN_AUDIO_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_AUDIO_DIR}")

    subdirs = [
        d
        for d in os.listdir(TRAIN_AUDIO_DIR)
        if os.path.isdir(os.path.join(TRAIN_AUDIO_DIR, d))
    ]

    for subdir in subdirs:
        dir_path = os.path.join(TRAIN_AUDIO_DIR, subdir)
        label = get_label_from_folder(subdir)

        # Glob all wav files
        files = glob.glob(os.path.join(dir_path, "*.wav"))

        for fpath in files:
            filename = os.path.basename(fpath)
            # Path relative to ./input
            rel_path = os.path.relpath(fpath, INPUT_DIR)

            speaker_id = None
            if label != "silence":
                # Filename format: <speaker_id>_nohash_<repetition>.wav
                parts = filename.split("_")
                if len(parts) > 1:
                    speaker_id = parts[0]
                else:
                    speaker_id = "unknown_speaker"

            records.append(
                {"filepath": rel_path, "label": label, "speaker_id": speaker_id}
            )

    return pd.DataFrame(records)


def parse_test_data():
    """Scans test directory and extracts metadata."""
    records = []
    if os.path.exists(TEST_AUDIO_DIR):
        files = glob.glob(os.path.join(TEST_AUDIO_DIR, "*.wav"))
        for fpath in files:
            rel_path = os.path.relpath(fpath, INPUT_DIR)
            records.append(
                {
                    "filepath": rel_path,
                    "label": "unknown",  # Placeholder for test
                    "speaker_id": None,
                }
            )
    return pd.DataFrame(records)


def split_data(df):
    """Splits data into train/val using Group Sampling on speaker_id."""
    # Separate silence (no speaker) and speech
    df_silence = df[df["label"] == "silence"].copy()
    df_speech = df[df["label"] != "silence"].copy()

    # --- Group Split for Speech ---
    unique_speakers = df_speech["speaker_id"].unique()

    # Shuffle speakers
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(unique_speakers)

    # Split speakers 80/20
    n_total = len(unique_speakers)
    n_train = int(n_total * 0.8)

    train_speakers = set(unique_speakers[:n_train])
    val_speakers = set(unique_speakers[n_train:])

    # Assign rows based on speaker
    train_speech = df_speech[df_speech["speaker_id"].isin(train_speakers)]
    val_speech = df_speech[df_speech["speaker_id"].isin(val_speakers)]

    # --- Random Split for Silence ---
    # Silence files are few and have no speaker ID, split randomly
    silence_indices = df_silence.index.to_numpy()
    rng.shuffle(silence_indices)

    n_silence_train = int(len(silence_indices) * 0.8)
    train_silence_idx = silence_indices[:n_silence_train]
    val_silence_idx = silence_indices[n_silence_train:]

    train_silence = df_silence.loc[train_silence_idx]
    val_silence = df_silence.loc[val_silence_idx]

    # --- Combine ---
    df_train = (
        pd.concat([train_speech, train_silence])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    df_val = (
        pd.concat([val_speech, val_silence])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    return df_train, df_val


def validate_datasets(df_train, df_val, df_test):
    """Performs integrity checks on the generated datasets."""
    print("\n=== Dataset Validation ===")

    # 1. Summary Statistics
    print(f"Train samples: {len(df_train)}")
    print(f"Val samples:   {len(df_val)}")
    print(f"Test samples:  {len(df_test)}")

    print("\nTrain Label Distribution:")
    print(df_train["label"].value_counts())

    print("\nVal Label Distribution:")
    print(df_val["label"].value_counts())

    # 2. Check File Existence (Random Sample)
    def check_paths(df, name):
        if len(df) == 0:
            return
        sample_size = min(1000, len(df))
        sample = df.sample(sample_size, random_state=RANDOM_STATE)

        missing = []
        for _, row in sample.iterrows():
            full_path = os.path.join(INPUT_DIR, row["filepath"])
            if not os.path.exists(full_path):
                missing.append(row["filepath"])

        missing_ratio = len(missing) / sample_size
        print(f"Missing file ratio ({name}): {missing_ratio:.4f}")

        if missing_ratio > 0.5:
            print("Example missing paths:", missing[:5])
            raise FileNotFoundError(
                f"Validation Failed: Too many missing files in {name} dataset."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Validation")
    check_paths(df_test, "Test")

    # 3. Verify Group Split (No Speaker Leakage)
    train_speakers = set(
        df_train[df_train["speaker_id"].notna()]["speaker_id"].unique()
    )
    val_speakers = set(df_val[df_val["speaker_id"].notna()]["speaker_id"].unique())

    intersection = train_speakers.intersection(val_speakers)
    print(f"Speaker intersection count: {len(intersection)}")

    if len(intersection) > 0:
        raise AssertionError(
            f"Validation Failed: Data leakage detected. {len(intersection)} speakers found in both Train and Validation sets."
        )

    print("=== Validation Passed ===")


def main():
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Parsing training data...")
    df_all_train = parse_training_data()

    print("Splitting data (Group Sampling by Speaker ID)...")
    df_train, df_val = split_data(df_all_train)

    print("Parsing test data...")
    df_test = parse_test_data()

    # Save to CSV
    print("Saving metadata...")
    df_train.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(METADATA_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # Validate
    validate_datasets(df_train, df_val, df_test)


if __name__ == "__main__":
    main()
