import os
import pandas as pd
import numpy as np
import torch
import shutil
from library.config import Config
from library.utils import seed_everything
from library.data_processing import get_processed_dataset, load_and_merge_tracking
from library.dataset import create_dataloader, ContactDataset
from library.model import KinematicFFN
from library.trainer import train_pipeline


def create_mini_dataset():
    """
    Creates a small subset of the training and test data in the working directory
    to allow for fast demonstration and testing of the pipeline.
    """
    print("Creating mini-dataset for demonstration...")

    # 1. Load a small sample of Train Metadata
    # We take the first 100 rows
    df_train_meta_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_mini_train_meta = df_train_meta_full.head(80).copy()
    df_mini_val_meta = df_train_meta_full.iloc[80:100].copy()  # Next 20 rows

    # 2. Load a small sample of Test Metadata
    df_test_meta_full = pd.read_csv(Config.TEST_METADATA_PATH)
    df_mini_test_meta = df_test_meta_full.head(20).copy()

    # 3. Create corresponding Mini Tracking Data
    # We need to ensure the tracking data covers the game_play and steps in our mini metadata
    # so the merge doesn't result in empty dataframes.

    # Collect all unique (game_play, step) pairs needed
    needed_keys = set()
    for df in [df_mini_train_meta, df_mini_val_meta]:
        df["game_play"] = df["game_play"].astype(str)
        for _, row in df.iterrows():
            needed_keys.add((row["game_play"], row["step"]))

    test_needed_keys = set()
    df_mini_test_meta["game_play"] = df_mini_test_meta["game_play"].astype(str)
    for _, row in df_mini_test_meta.iterrows():
        test_needed_keys.add((row["game_play"], row["step"]))

    # Load full tracking data (it's around 1.2M rows, manageable to filter in memory for this task)
    print("Filtering training tracking data...")
    df_track_full = pd.read_csv(Config.TRAIN_TRACKING_PATH)
    df_track_full["game_play"] = df_track_full["game_play"].astype(str)

    # Filter tracking data
    # We create a mask based on game_play and step
    # To speed up, we can just filter by game_play first
    needed_plays = set(k[0] for k in needed_keys)
    df_mini_track = df_track_full[df_track_full["game_play"].isin(needed_plays)].copy()

    if df_mini_track.empty:
        raise ValueError(
            "Mini training tracking data is empty! Check game_play matching."
        )

    # Further filter by step to keep file size minimal (optional, but good for strictness)
    # For simplicity in this demo, keeping all steps for the relevant plays is sufficient
    # and safer for context.

    # Do the same for Test Tracking
    print("Filtering test tracking data...")
    df_test_track_full = pd.read_csv(Config.TEST_TRACKING_PATH)
    df_test_track_full["game_play"] = df_test_track_full["game_play"].astype(str)
    test_needed_plays = set(k[0] for k in test_needed_keys)
    df_mini_test_track = df_test_track_full[
        df_test_track_full["game_play"].isin(test_needed_plays)
    ].copy()

    # 4. Save Mini Files to Working Directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    mini_train_meta_path = os.path.join(Config.WORKING_DIR, "mini_train_metadata.csv")
    mini_val_meta_path = os.path.join(Config.WORKING_DIR, "mini_val_metadata.csv")
    mini_test_meta_path = os.path.join(Config.WORKING_DIR, "mini_test_metadata.csv")
    mini_train_track_path = os.path.join(Config.WORKING_DIR, "mini_train_tracking.csv")
    mini_test_track_path = os.path.join(Config.WORKING_DIR, "mini_test_tracking.csv")

    df_mini_train_meta.to_csv(mini_train_meta_path, index=False)
    df_mini_val_meta.to_csv(mini_val_meta_path, index=False)
    df_mini_test_meta.to_csv(mini_test_meta_path, index=False)
    df_mini_track.to_csv(mini_train_track_path, index=False)
    df_mini_test_track.to_csv(mini_test_track_path, index=False)

    print("Mini-dataset created successfully.")

    return {
        "train_meta": mini_train_meta_path,
        "val_meta": mini_val_meta_path,
        "test_meta": mini_test_meta_path,
        "train_track": mini_train_track_path,
        "test_track": mini_test_track_path,
    }


def demonstrate_components(paths):
    """
    Demonstrates individual component usage: Data Processing, Dataset, Model.
    """
    print("\n--- Demonstrating Individual Components ---")

    # 1. Data Processing
    print("1. Processing Training Data...")
    # We force load_cached_data=False to ensure the logic runs
    X, y, ids, scaler = get_processed_dataset(
        mode="train",
        metadata_path=paths["train_meta"],
        tracking_path=paths["train_track"],
        load_cached_data=False,
    )

    print(f"   Processed Feature Shape: {X.shape}")
    print(f"   Target Shape: {y.shape}")

    # Assertions
    assert X.shape[0] == y.shape[0], "Feature and target row counts must match."
    assert X.shape[1] == len(
        Config.FEATURES
    ), f"Feature count must match Config.FEATURES ({len(Config.FEATURES)})."
    assert isinstance(scaler, object), "Scaler object should be returned."

    # 2. Dataset and DataLoader
    print("2. Creating Dataset and DataLoader...")
    loader = create_dataloader(X, y, batch_size=16, shuffle=True)

    # Fetch one batch
    features_batch, targets_batch = next(iter(loader))
    print(f"   Batch Feature Shape: {features_batch.shape}")
    print(f"   Batch Target Shape: {targets_batch.shape}")

    assert features_batch.shape == (16, len(Config.FEATURES))
    assert targets_batch.shape == (16, 1)

    # 3. Model Instantiation and Forward Pass
    print("3. Model Initialization and Forward Pass...")
    input_dim = len(Config.FEATURES)
    model = KinematicFFN(input_dim=input_dim, hidden_layers=[32, 16])

    # Move to CPU for this quick test
    model.cpu()
    output = model(features_batch)

    print(f"   Model Output Shape: {output.shape}")
    assert output.shape == (16, 1), "Model output shape mismatch."
    assert torch.all(
        (output >= 0) & (output <= 1)
    ), "Sigmoid output must be between 0 and 1."

    return scaler


def run_full_pipeline_demo(paths):
    """
    Runs the full training pipeline using the patched configuration.
    """
    print("\n--- Running Full Training Pipeline ---")

    # Monkey-patch Config to use our mini datasets
    # The trainer module uses Config.*_PATH constants directly
    Config.TRAIN_METADATA_PATH = paths["train_meta"]
    Config.VAL_METADATA_PATH = paths["val_meta"]
    Config.TEST_METADATA_PATH = paths["test_meta"]

    # Note: The trainer uses TRAIN_TRACKING_PATH for both train and val
    Config.TRAIN_TRACKING_PATH = paths["train_track"]
    Config.TEST_TRACKING_PATH = paths["test_track"]

    # Use a separate cache dir for the demo to avoid messing with real experiments
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")

    # Run pipeline
    # We use very few epochs and small batch size for speed
    model, threshold, mcc = train_pipeline(
        epochs=2,
        batch_size=32,
        learning_rate=0.01,
        patience=1,
        load_cached_data=False,  # Force re-processing with patched paths
    )

    print(f"\nPipeline Completed.")
    print(f"Best MCC: {mcc:.4f}")
    print(f"Optimal Threshold: {threshold:.4f}")

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission File Generated at: {Config.SUBMISSION_PATH}")
        print(df_sub.head())

        # Assertions
        assert "contact_id" in df_sub.columns
        assert "contact" in df_sub.columns
        assert len(df_sub) > 0
        assert df_sub["contact"].isin([0, 1]).all(), "Predictions must be binary."
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup_directories()

    # 2. Create Data
    paths = create_mini_dataset()

    # 3. Demonstrate Components (Unit Tests)
    scaler = demonstrate_components(paths)

    # 4. Demonstrate Full Pipeline (Integration Test)
    run_full_pipeline_demo(paths)

    print("\nAll demonstrations and assertions passed successfully.")
