import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Import library components
from library.config import Config
from library.utils import seed_everything, optimize_threshold
from library.feature_engineering import (
    prepare_wide_dataset,
    impute_ground_physics,
    compute_relative_physics,
)
from library.dataset import get_dataloader
from library.model import WideResNetMLP
from library.loss import FocalLoss


def setup_demo_config():
    """
    Overrides Config parameters for a quick demo run.
    """
    print(">>> Setting up demo configuration...")
    Config.WORKING_DIR = "./working/demo_output"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Reduce compute load
    Config.BATCH_SIZE = 128
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WINDOW_SIZE = 2  # Smaller window for faster feature gen

    # Paths for demo artifacts
    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "demo_scaler.joblib")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")


def demonstrate_feature_engineering():
    """
    Demonstrates and validates the feature engineering pipeline on a subset of data.
    """
    print("\n>>> Demonstrating Feature Engineering...")

    # 1. Load Subset of Metadata
    # We pick the first 500 rows to keep it fast
    df_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv")).head(500)

    # 2. Load Relevant Tracking Data
    # We only need tracking for the game_plays present in our metadata subset
    unique_games = df_meta["game_play"].unique()
    df_tracking = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "train_player_tracking.csv")
    )
    df_tracking = df_tracking[df_tracking["game_play"].isin(unique_games)].copy()

    print(f"  Subset Metadata Shape: {df_meta.shape}")
    print(f"  Subset Tracking Shape: {df_tracking.shape}")

    # 3. Run Pipeline
    # prepare_wide_dataset handles lagging, merging, ground imputation, and relative physics
    X, y = prepare_wide_dataset(df_meta, df_tracking, is_train=True)

    # 4. Validations
    print("  Validating feature outputs...")

    # Check dimensions
    assert len(X) == len(
        df_meta
    ), f"Feature rows {len(X)} != Metadata rows {len(df_meta)}"
    assert len(y) == len(df_meta), "Target length mismatch"

    # Check for Lag Columns
    # With WINDOW_SIZE=2, we expect lags -2, -1, 0, 1, 2
    expected_lag_col = "speed_1_lag-2"
    assert (
        expected_lag_col in X.columns
    ), f"Missing expected lag column: {expected_lag_col}"

    # Check Relative Physics
    # 'log_dist_lag0' should exist
    assert (
        "log_dist_lag0" in X.columns
    ), "Missing relative physics column: log_dist_lag0"

    # Check Ground Imputation Logic
    # We manually verify if we have any ground rows in this subset
    # In the metadata, nfl_player_id_2 == 'G'
    ground_rows = df_meta["nfl_player_id_2"] == "G"
    if ground_rows.any():
        # Get indices of ground rows
        ground_indices = ground_rows[ground_rows].index

        # Check a specific ground feature, e.g., speed_2_lag0 should be 0
        # Note: X index aligns with df_meta index because prepare_wide_dataset preserves order/length of metadata
        ground_speeds = X.loc[ground_indices, "speed_2_lag0"]
        assert (
            ground_speeds == 0
        ).all(), "Ground imputation failed: Ground speed is not 0"
        print("  Ground imputation logic verified.")

    # Check for NaNs (Pipeline should handle them, usually filling with 0 or keeping valid)
    # The provided prepare_wide_dataset fills NaNs with 0.0 at the end
    assert not X.isnull().values.any(), "Features contain NaNs"

    print("  Feature Engineering successful. Shape:", X.shape)
    return X, y


def demonstrate_model_training(X_df, y_array):
    """
    Demonstrates model instantiation, data loading, and a training step.
    """
    print("\n>>> Demonstrating Model Training Components...")

    # 1. Preprocessing (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # 2. DataLoader
    # Convert to float32 is handled inside Dataset, but input to Dataset is numpy array
    loader = get_dataloader(
        features=X_scaled,
        targets=y_array,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch to verify
    features_batch, targets_batch = next(iter(loader))
    print(
        f"  Batch Shapes - Features: {features_batch.shape}, Targets: {targets_batch.shape}"
    )

    assert features_batch.dtype == torch.float32
    assert targets_batch.dtype == torch.float32

    # 3. Model Initialization
    input_dim = X_scaled.shape[1]
    model = WideResNetMLP(input_dim=input_dim, hidden_dim=64, num_blocks=1).to(
        Config.DEVICE
    )
    print(f"  Model initialized on {Config.DEVICE}")

    # 4. Loss Function
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # 5. Training Step Simulation
    model.train()

    # Forward Pass
    logits = model(features_batch.to(Config.DEVICE)).squeeze(1)

    # Loss Calculation
    loss = criterion(logits, targets_batch.to(Config.DEVICE))

    print(f"  Initial Loss: {loss.item():.4f}")

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("  Backward pass and optimizer step successful.")

    # Check if weights changed (simple check on input projection layer)
    # We can't easily check 'changed' without a copy, but successful execution implies gradients flowed.
    assert not torch.isnan(loss), "Loss is NaN"

    return model, scaler


def demonstrate_inference(model, scaler):
    """
    Demonstrates the inference pipeline using test metadata.
    """
    print("\n>>> Demonstrating Inference Pipeline...")

    # 1. Load Subset of Test Metadata
    df_test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv")).head(100)

    # 2. Load Tracking
    # Note: Test tracking might be needed for the specific game_plays
    unique_games = df_test_meta["game_play"].unique()
    df_tracking = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "test_player_tracking.csv")
    )
    # Filter to ensure we have data for the subset
    df_tracking = df_tracking[df_tracking["game_play"].isin(unique_games)].copy()

    if df_tracking.empty:
        print(
            "  Warning: No matching tracking data for test subset. Skipping feature gen validation."
        )
        return

    # 3. Generate Features
    # is_train=False returns (X, merged_df_with_ids)
    X_test_df, merged_test = prepare_wide_dataset(
        df_test_meta, df_tracking, is_train=False
    )

    # 4. Scale
    X_test_scaled = scaler.transform(X_test_df)

    # 5. Predict
    model.eval()
    with torch.no_grad():
        # Convert to tensor
        x_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(Config.DEVICE)
        logits = model(x_tensor).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    # 6. Verify Output
    assert len(probs) == len(df_test_meta)
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    print(f"  Inference successful. Generated {len(probs)} predictions.")
    print(f"  Sample Probabilities: {probs[:5]}")

    # Demonstrate Threshold Optimization utility
    # Create dummy ground truth for demonstration since test set has no labels
    dummy_y_true = np.random.randint(0, 2, size=len(probs))
    best_thresh, best_mcc = optimize_threshold(dummy_y_true, probs)
    print(
        f"  Dummy Threshold Optimization -> Best Thresh: {best_thresh}, MCC: {best_mcc}"
    )


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Setup
    setup_demo_config()

    try:
        # Step 1: Feature Engineering
        X, y = demonstrate_feature_engineering()

        # Step 2: Training Loop
        model, scaler = demonstrate_model_training(X, y)

        # Step 3: Inference
        demonstrate_inference(model, scaler)

        print("\n>>> All demonstrations completed successfully.")

    except Exception as e:
        print(f"\n!!! Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
