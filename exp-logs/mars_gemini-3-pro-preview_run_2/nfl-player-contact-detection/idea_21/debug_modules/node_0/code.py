import os
import shutil
import numpy as np
import pandas as pd
import torch
import joblib

# Import library components
from library.config import Config
from library.feature_engineering import generate_dataset
from library.dataset import prepare_dataloaders
from library.model import GRVCNet
from library.loss import FocalLoss
from library.trainer import run_training

# Set fixed seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def setup_demo_environment():
    """
    Sets up a lightweight environment for demonstration by:
    1. Creating a temporary working directory.
    2. Creating subset metadata files (1 game_play each) to speed up processing.
    3. Overriding Config parameters to use these subsets and run fast.
    """
    print("--- Setting up Demo Environment ---")

    # Define paths
    demo_work_dir = "./working/demo_execution"
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)
    os.makedirs(demo_work_dir, exist_ok=True)

    # Override Config Global Settings
    Config.WORKING_DIR = demo_work_dir
    Config.OUTPUT_DIR = demo_work_dir
    Config.SUBMISSION_PATH = os.path.join(demo_work_dir, "submission.csv")
    Config.SCALER_PATH = os.path.join(demo_work_dir, "scaler.joblib")

    # Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 64
    Config.PATIENCE = 1

    # Create Subset Metadata
    # We read the original metadata and pick the first game_play for each split

    # Train Metadata
    df_train_orig = pd.read_csv("./metadata/train.csv")
    train_gp = df_train_orig["game_play"].unique()[0]
    df_train_sub = df_train_orig[df_train_orig["game_play"] == train_gp].copy()
    new_train_meta = os.path.join(demo_work_dir, "train_meta.csv")
    df_train_sub.to_csv(new_train_meta, index=False)
    Config.METADATA_TRAIN = new_train_meta

    # Validation Metadata
    df_val_orig = pd.read_csv("./metadata/validation.csv")
    val_gp = df_val_orig["game_play"].unique()[0]
    df_val_sub = df_val_orig[df_val_orig["game_play"] == val_gp].copy()
    new_val_meta = os.path.join(demo_work_dir, "val_meta.csv")
    df_val_sub.to_csv(new_val_meta, index=False)
    Config.METADATA_VAL = new_val_meta

    # Test Metadata
    df_test_orig = pd.read_csv("./metadata/test.csv")
    test_gp = df_test_orig["game_play"].unique()[0]
    df_test_sub = df_test_orig[df_test_orig["game_play"] == test_gp].copy()
    new_test_meta = os.path.join(demo_work_dir, "test_meta.csv")
    df_test_sub.to_csv(new_test_meta, index=False)
    Config.METADATA_TEST = new_test_meta

    print(f"Subset metadata created in {demo_work_dir}")
    print(
        f"Train samples: {len(df_train_sub)}, Val samples: {len(df_val_sub)}, Test samples: {len(df_test_sub)}"
    )


def demonstrate_feature_engineering():
    """
    Demonstrates the feature engineering pipeline.
    """
    print("\n--- Demonstrating Feature Engineering ---")

    # Generate features for the training subset
    # load_cached_data=False ensures we actually run the logic
    df_features = generate_dataset(mode="train", load_cached_data=False)

    # Validations
    assert isinstance(df_features, pd.DataFrame), "Output is not a DataFrame"
    assert not df_features.empty, "Feature DataFrame is empty"
    assert "contact" in df_features.columns, "Target column 'contact' missing"

    # Check for specific engineered features
    # Kinematic Lag
    assert (
        "p1_speed_lag_0" in df_features.columns
    ), "Kinematic feature 'p1_speed_lag_0' missing"
    # Relative Physics
    assert (
        "log_dist_lag_0" in df_features.columns
    ), "Relative feature 'log_dist_lag_0' missing"
    # Visuals
    assert "v_width" in df_features.columns, "Visual feature 'v_width' missing"

    print(f"Feature Engineering successful. Output shape: {df_features.shape}")
    return df_features


def demonstrate_model_components(df_sample):
    """
    Demonstrates model instantiation, forward pass, and loss calculation.
    """
    print("\n--- Demonstrating Model Components ---")

    # Identify feature dimensions dynamically
    all_cols = df_sample.columns
    vis_cols = [c for c in all_cols if c.startswith("v_")]
    exclude = [
        "contact_id",
        "game_play",
        "contact",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
    ] + vis_cols
    kin_cols = [c for c in all_cols if c not in exclude]

    kin_dim = len(kin_cols)
    vis_dim = len(vis_cols)
    print(f"Detected Input Dims -> Kinematic: {kin_dim}, Visual: {vis_dim}")

    # 1. Instantiate Model
    device = torch.device("cpu")  # CPU is sufficient for demo
    model = GRVCNet(kin_dim, vis_dim, Config).to(device)

    # 2. Create Dummy Batch
    batch_size = 4
    x_kin = torch.randn(batch_size, kin_dim).to(device)
    x_vis = torch.randn(batch_size, vis_dim).to(device)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0]).to(device)

    # 3. Forward Pass
    logits = model(x_kin, x_vis)

    # Validate Output Shape
    assert logits.shape == (
        batch_size,
    ), f"Expected output shape ({batch_size},), got {logits.shape}"

    # 4. Loss Calculation
    criterion = FocalLoss()
    loss = criterion(logits, targets)

    # Validate Loss
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print(
        f"Model forward pass and loss calculation successful. Loss: {loss.item():.4f}"
    )


def demonstrate_full_pipeline():
    """
    Runs the complete training and inference pipeline using library.trainer.
    """
    print("\n--- Demonstrating Full Training Pipeline ---")

    # Run training (includes data loading, training loop, validation, threshold opt, and submission)
    # We use cached data=True because generate_dataset was called in the previous step
    # and saved to the demo directory.
    run_training(
        load_cached_data=True, batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS
    )

    # Validate Artifacts
    # 1. Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated"
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_sub.columns) == [
        "contact_id",
        "contact",
    ], "Submission columns are incorrect"
    assert len(df_sub) > 0, "Submission file is empty"

    # 2. Scaler
    assert os.path.exists(Config.SCALER_PATH), "Scaler joblib file was not saved"

    # 3. Best Model (might not exist if validation never improved, but with 1 epoch it usually saves once)
    # However, run_training logic saves if val_mcc > -1. Initial best is -1.0.
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(model_path):
        print("Best model checkpoint verified.")
    else:
        print(
            "Note: Best model not saved (likely due to metric performance in 1 epoch), but pipeline finished."
        )

    print(
        f"Pipeline finished successfully. Submission saved to {Config.SUBMISSION_PATH}"
    )


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Feature Engineering
    df_train_sample = demonstrate_feature_engineering()

    # 3. Model Logic
    demonstrate_model_components(df_train_sample)

    # 4. Full Pipeline
    demonstrate_full_pipeline()

    print("\nAll demonstrations completed successfully.")
