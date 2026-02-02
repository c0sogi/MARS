import pandas as pd
import numpy as np
import os

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
RANDOM_STATE = 42
TRAIN_RATIO = 0.8


def generate_metadata():
    # Create metadata directory
    os.makedirs(METADATA_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Training and Validation Data
    # ---------------------------------------------------------
    print("Loading train_labels.csv...")
    train_labels_path = os.path.join(INPUT_DIR, "train_labels.csv")
    df_full = pd.read_csv(train_labels_path)

    # Get unique plays for Group Splitting
    unique_plays = df_full["game_play"].unique()

    # Shuffle plays with fixed random state
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(unique_plays)

    # Split plays
    n_train = int(len(unique_plays) * TRAIN_RATIO)
    train_plays = set(unique_plays[:n_train])
    val_plays = set(unique_plays[n_train:])

    print(f"Total plays: {len(unique_plays)}")
    print(f"Train plays: {len(train_plays)}")
    print(f"Val plays: {len(val_plays)}")

    # Create masks
    train_mask = df_full["game_play"].isin(train_plays)
    val_mask = df_full["game_play"].isin(val_plays)

    df_train = df_full[train_mask].copy()
    df_val = df_full[val_mask].copy()

    # Function to add video paths
    def add_video_paths(df, source_folder):
        # Vectorized string concatenation for performance
        base = f"./input/{source_folder}/" + df["game_play"]
        df["video_path_endzone"] = base + "_Endzone.mp4"
        df["video_path_sideline"] = base + "_Sideline.mp4"
        df["video_path_all29"] = base + "_All29.mp4"
        return df

    # Add paths (Train/Val videos are in input/train)
    df_train = add_video_paths(df_train, "train")
    df_val = add_video_paths(df_val, "train")

    # Save to metadata
    print("Saving train_metadata.csv...")
    df_train.to_csv(os.path.join(METADATA_DIR, "train_metadata.csv"), index=False)

    print("Saving val_metadata.csv...")
    df_val.to_csv(os.path.join(METADATA_DIR, "val_metadata.csv"), index=False)

    # ---------------------------------------------------------
    # 2. Process Test Data
    # ---------------------------------------------------------
    print("Processing test data from sample_submission.csv...")
    sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    df_test = pd.read_csv(sample_sub_path)

    # Parse contact_id to extract features
    # Format: game_key_play_id_step_p1_p2
    # We split by '_'
    # Note: game_key is typically 5 digits, play_id variable.
    # We rely on the structure provided in the description.
    splits = df_test["contact_id"].str.split("_", expand=True)

    # Construct columns
    # game_play = splits[0] + "_" + splits[1]
    df_test["game_play"] = splits[0] + "_" + splits[1]

    # step is at index 2
    df_test["step"] = splits[2].astype(int)

    # player1 is at index 3
    df_test["nfl_player_id_1"] = splits[3].astype(int)

    # player2 is at index 4 (can be 'G', so keep as object)
    df_test["nfl_player_id_2"] = splits[4]

    # Add paths (Test videos are in input/test)
    df_test = add_video_paths(df_test, "test")

    # Save to metadata
    print("Saving test_metadata.csv...")
    df_test.to_csv(os.path.join(METADATA_DIR, "test_metadata.csv"), index=False)

    # ---------------------------------------------------------
    # 3. Verification
    # ---------------------------------------------------------
    print("\n--- Verification ---")

    # Reload datasets
    df_train_check = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    df_val_check = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    df_test_check = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # Summary Stats
    print(f"Train Rows: {len(df_train_check)}")
    print(f"Val Rows: {len(df_val_check)}")
    print(f"Test Rows: {len(df_test_check)}")

    # Verify Split (No overlap in game_play)
    train_plays_check = set(df_train_check["game_play"].unique())
    val_plays_check = set(df_val_check["game_play"].unique())

    overlap = train_plays_check.intersection(val_plays_check)
    if overlap:
        raise AssertionError(
            f"Data leakage detected! {len(overlap)} plays found in both train and val."
        )
    print("Split verification passed: No overlap between train and val plays.")

    # Verify File Paths
    def check_random_paths(df, name):
        if df.empty:
            return

        # Collect all video path columns
        path_cols = ["video_path_endzone", "video_path_sideline", "video_path_all29"]

        # Sample 1000 random rows (or all if less than 1000)
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size, random_state=RANDOM_STATE)

        # Flatten to get a list of paths from the sample
        # We will check one random view per sampled row to keep check count to ~1000
        # Or we can check all. Let's check a random selection of 1000 paths from the available columns.

        all_paths = sample_df[path_cols].values.flatten()

        # Select 1000 paths to check
        rng_check = np.random.RandomState(RANDOM_STATE)
        paths_to_check = rng_check.choice(
            all_paths, size=min(1000, len(all_paths)), replace=False
        )

        missing_count = 0
        missing_samples = []

        for p in paths_to_check:
            if not os.path.exists(p):
                missing_count += 1
                if len(missing_samples) < 5:
                    missing_samples.append(p)

        ratio = missing_count / len(paths_to_check)
        print(
            f"[{name}] Missing file ratio: {ratio:.4f} ({missing_count}/{len(paths_to_check)})"
        )

        if ratio > 0.5:
            print("Sample missing paths:", missing_samples)
            raise FileNotFoundError(
                f"Missing file ratio ({ratio}) exceeds 0.5 for {name} dataset."
            )

    check_random_paths(df_train_check, "Train")
    check_random_paths(df_val_check, "Val")
    check_random_paths(df_test_check, "Test")

    print("All checks passed successfully.")


if __name__ == "__main__":
    generate_metadata()
