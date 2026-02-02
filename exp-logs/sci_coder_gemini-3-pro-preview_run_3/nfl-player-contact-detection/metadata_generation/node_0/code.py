import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import GroupShuffleSplit

# Configuration
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def generate_metadata():
    # Ensure metadata directory exists
    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading datasets...")
    train_labels_path = os.path.join(INPUT_DIR, "train_labels.csv")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Load Train Labels
    # train_labels.csv columns: contact_id, game_play, nfl_player_id_1, nfl_player_id_2, step, datetime, contact
    df_train_full = pd.read_csv(train_labels_path)

    # Load Sample Submission (Test)
    # sample_submission.csv columns: contact_id, contact
    df_test = pd.read_csv(sample_sub_path)

    # --- Process Test Data ---
    # We need to parse contact_id to get game_play and other fields to match train schema where possible
    # contact_id format: game_key_play_id_step_player1_player2
    # Example: 58168_003392_0_38590_43854
    print("Processing test data...")

    # Vectorized parsing of contact_id for test set
    # We split by underscore.
    # The first two parts form game_play (game_key + play_id)
    # The third is step
    # The fourth is player1
    # The fifth is player2

    # Using str.split with expand=True is convenient
    split_data = df_test["contact_id"].str.split("_", expand=True)

    # Assign columns
    # Note: split_data columns will be 0, 1, 2, 3, 4
    df_test["game_play"] = split_data[0] + "_" + split_data[1]
    df_test["step"] = split_data[2].astype(int)
    df_test["nfl_player_id_1"] = split_data[
        3
    ]  # Keep as string/object to match 'G' possibility or consistent IDs
    df_test["nfl_player_id_2"] = split_data[4]

    # --- Add Video Paths ---
    # Paths are relative to ./input
    # Train videos: train/{game_play}_Sideline.mp4, train/{game_play}_Endzone.mp4
    # Test videos: test/{game_play}_Sideline.mp4, test/{game_play}_Endzone.mp4

    def add_video_paths(df, folder_name):
        df["video_path_sideline"] = (
            folder_name + "/" + df["game_play"] + "_Sideline.mp4"
        )
        df["video_path_endzone"] = folder_name + "/" + df["game_play"] + "_Endzone.mp4"
        # All29 view is also mentioned in description, adding it as well
        df["video_path_all29"] = folder_name + "/" + df["game_play"] + "_All29.mp4"
        return df

    df_train_full = add_video_paths(df_train_full, "train")
    df_test = add_video_paths(df_test, "test")

    # --- Split Train/Validation ---
    print("Splitting train/validation...")
    # Requirement: Group Sampling on game_play, 80:20 split, Random State 42
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    # We split based on the groups (game_play)
    train_idx, val_idx = next(
        gss.split(df_train_full, groups=df_train_full["game_play"])
    )

    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    # --- Save Metadata ---
    print("Saving metadata...")
    train_save_path = os.path.join(METADATA_DIR, "train.csv")
    val_save_path = os.path.join(METADATA_DIR, "validation.csv")
    test_save_path = os.path.join(METADATA_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    return df_train, df_val, df_test


def validate_metadata(df_train, df_val, df_test):
    print("\n--- Validating Metadata ---")

    # 1. Print Summary Statistics
    print(f"Train set shape: {df_train.shape}")
    print(f"Validation set shape: {df_val.shape}")
    print(f"Test set shape: {df_test.shape}")

    print(f"Train unique plays: {df_train['game_play'].nunique()}")
    print(f"Validation unique plays: {df_val['game_play'].nunique()}")
    print(f"Test unique plays: {df_test['game_play'].nunique()}")

    train_pos_rate = df_train["contact"].mean()
    val_pos_rate = df_val["contact"].mean()
    print(f"Train contact rate: {train_pos_rate:.4f}")
    print(f"Validation contact rate: {val_pos_rate:.4f}")

    # 2. Check File Paths
    print("\nChecking file paths...")

    def check_paths(df, name):
        # Sample 1000 paths (or all if less than 1000)
        n_sample = min(1000, len(df))
        sample = df.sample(n=n_sample, random_state=RANDOM_STATE)

        # Check Sideline paths
        missing_count = 0
        missing_samples = []

        for _, row in sample.iterrows():
            # We check one view type primarily, but let's check Sideline as it's standard
            rel_path = row["video_path_sideline"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        ratio = missing_count / n_sample
        print(f"Missing file ratio for {name} (Sideline): {ratio:.4f}")

        if ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"More than 50% of file paths missing in {name} metadata."
            )

    check_paths(df_train, "Train")
    check_paths(df_val, "Validation")
    check_paths(df_test, "Test")

    # 3. Verify Validation Split Requirements
    print("\nVerifying split requirements...")

    # Check overlap of groups
    train_plays = set(df_train["game_play"].unique())
    val_plays = set(df_val["game_play"].unique())

    overlap = train_plays.intersection(val_plays)
    if overlap:
        raise AssertionError(
            f"Data Leakage detected! {len(overlap)} plays found in both train and validation sets."
        )
    else:
        print("Success: No play overlap between train and validation.")

    # Check split ratio roughly
    total_plays = df_train["game_play"].nunique() + df_val["game_play"].nunique()
    actual_val_ratio = df_val["game_play"].nunique() / total_plays
    print(f"Actual Validation Split Ratio (by play count): {actual_val_ratio:.4f}")

    # Allow some variance due to play size differences, but it should be close to 0.2 in terms of groups
    if not (0.15 < actual_val_ratio < 0.25):
        print(
            "Warning: Split ratio deviates significantly from 0.2. This might be due to small dataset size or uneven play lengths."
        )

    print("\nMetadata generation and validation completed successfully.")


if __name__ == "__main__":
    try:
        train_df, val_df, test_df = generate_metadata()
        validate_metadata(train_df, val_df, test_df)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
