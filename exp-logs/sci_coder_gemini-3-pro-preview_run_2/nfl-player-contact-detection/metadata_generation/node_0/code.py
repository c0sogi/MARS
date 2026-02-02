import pandas as pd
import numpy as np
import os
import glob


def generate_metadata():
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    RANDOM_STATE = 42

    os.makedirs(METADATA_DIR, exist_ok=True)

    print("Loading raw data...")
    train_labels = pd.read_csv(os.path.join(INPUT_DIR, "train_labels.csv"))
    sample_submission = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

    # --- Process Training Data ---
    print("Processing training data...")

    # Get unique groups for splitting
    groups = train_labels["game_play"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(groups)

    split_idx = int(len(groups) * 0.8)
    train_groups = set(groups[:split_idx])
    val_groups = set(groups[split_idx:])

    # Create split masks
    is_train = train_labels["game_play"].isin(train_groups)
    is_val = train_labels["game_play"].isin(val_groups)

    df_train = train_labels[is_train].copy()
    df_val = train_labels[is_val].copy()

    # Helper to add video paths
    def add_video_paths(df, folder_name):
        # Video files are named {game_play}_{view}.mp4
        # We assume the standard views exist: Endzone, Sideline, All29
        # Paths are relative to ./input

        # Vectorized path creation
        gp = df["game_play"]
        df["path_endzone"] = folder_name + "/" + gp + "_Endzone.mp4"
        df["path_sideline"] = folder_name + "/" + gp + "_Sideline.mp4"
        df["path_all29"] = folder_name + "/" + gp + "_All29.mp4"
        return df

    df_train = add_video_paths(df_train, "train")
    df_val = add_video_paths(df_val, "train")

    # --- Process Test Data ---
    print("Processing test data...")

    # sample_submission only has contact_id and contact. We need to parse contact_id.
    # contact_id format: game_key_play_id_step_player1_player2
    # But game_key is 5 digits, play_id is variable (usually 6 digits padded).
    # Let's split by underscore.

    # Using vectorized string split
    # Example: 58168_003392_0_38590_43854
    # parts: [58168, 003392, 0, 38590, 43854]
    # game_play = 58168_003392

    # We need to handle the variable length carefully, but the format is consistent:
    # {game_key}_{play_id}_{step}_{p1}_{p2}
    # game_play is {game_key}_{play_id}

    split_data = sample_submission["contact_id"].str.split("_", expand=True)

    # Construct columns
    # game_play is col 0 + '_' + col 1
    df_test = sample_submission.copy()
    df_test["game_play"] = split_data[0] + "_" + split_data[1]
    df_test["step"] = split_data[2].astype(int)
    df_test["nfl_player_id_1"] = split_data[
        3
    ]  # Keep as object/string to handle potential inconsistencies or just consistency with labels
    df_test["nfl_player_id_2"] = split_data[4]

    # Try to add datetime from test_player_tracking if possible
    # Loading test_player_tracking to map (game_play, step) -> datetime
    try:
        test_tracking = pd.read_csv(
            os.path.join(INPUT_DIR, "test_player_tracking.csv"),
            usecols=["game_play", "step", "datetime"],
        )
        # Drop duplicates to get unique time per step
        time_map = test_tracking.drop_duplicates(subset=["game_play", "step"])[
            ["game_play", "step", "datetime"]
        ]
        df_test = df_test.merge(time_map, on=["game_play", "step"], how="left")
    except Exception as e:
        print(f"Warning: Could not merge datetime for test set: {e}")
        df_test["datetime"] = np.nan

    df_test = add_video_paths(df_test, "test")

    # --- Save Metadata ---
    print("Saving metadata...")
    df_train.to_csv(os.path.join(METADATA_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(METADATA_DIR, "validation.csv"), index=False)
    df_test.to_csv(os.path.join(METADATA_DIR, "test.csv"), index=False)

    # --- Verification ---
    print("Verifying datasets...")

    # Reload
    meta_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    meta_val = pd.read_csv(os.path.join(METADATA_DIR, "validation.csv"))
    meta_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Print Statistics
    print("\n=== Dataset Statistics ===")
    print(f"Train samples: {len(meta_train)}")
    print(f"Validation samples: {len(meta_val)}")
    print(f"Test samples: {len(meta_test)}")

    print(f"Train unique game_plays: {meta_train['game_play'].nunique()}")
    print(f"Validation unique game_plays: {meta_val['game_play'].nunique()}")
    print(f"Test unique game_plays: {meta_test['game_play'].nunique()}")

    print(f"Train contact ratio: {meta_train['contact'].mean():.4f}")
    print(f"Validation contact ratio: {meta_val['contact'].mean():.4f}")

    # 2. Check Split Logic
    train_gps = set(meta_train["game_play"].unique())
    val_gps = set(meta_val["game_play"].unique())

    intersection = train_gps.intersection(val_gps)
    if len(intersection) > 0:
        raise AssertionError(
            f"Data Leakage Detected! {len(intersection)} game_plays found in both train and validation."
        )
    else:
        print(
            "Split verification passed: No overlap between train and validation groups."
        )

    # 3. Check File Paths
    print("\nChecking file path existence...")

    def check_paths(df, name):
        # Columns to check
        path_cols = ["path_endzone", "path_sideline", "path_all29"]

        # Sample 1000 paths total across these columns
        # We'll sample 1000 rows, then pick one random column for each
        if len(df) > 1000:
            sample = df.sample(1000, random_state=RANDOM_STATE)
        else:
            sample = df

        missing_count = 0
        total_checked = 0
        missing_samples = []

        for _, row in sample.iterrows():
            # Check all 3 paths for the row to be thorough, or just 1?
            # Requirement: "check 1000 relative file paths randomly selected"
            # Let's pick one random column per row to get 1000 paths.
            col = np.random.choice(path_cols)
            rel_path = row[col]
            full_path = os.path.join(INPUT_DIR, rel_path)

            total_checked += 1
            if not os.path.exists(full_path):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(rel_path)

        ratio = missing_count / total_checked if total_checked > 0 else 0
        print(
            f"[{name}] Missing file ratio: {ratio:.4f} ({missing_count}/{total_checked})"
        )

        if ratio > 0.5:
            print("Sample missing paths:")
            for p in missing_samples:
                print(f"  {p}")
            raise FileNotFoundError(
                f"Too many missing files in {name} dataset! Ratio: {ratio}"
            )

    check_paths(meta_train, "Train")
    check_paths(meta_val, "Validation")
    check_paths(meta_test, "Test")

    print("\nMetadata generation and verification complete.")


if __name__ == "__main__":
    generate_metadata()
