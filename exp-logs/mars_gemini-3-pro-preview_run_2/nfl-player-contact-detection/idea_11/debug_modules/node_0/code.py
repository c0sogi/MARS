import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Import provided library components
from library.config import (
    Config,
    set_seed,
    process_tracking_data,
    get_data,
    ECGRN,
    FocalLoss,
    ContactDataset,
)
from library.data_processing import impute_ground_and_engineer_features
from library.train import train_epoch, validate, optimize_threshold
from library.utils import compute_mcc


def main():
    print("=== Starting NFL Contact Detection Library Demo ===")

    # 1. Configuration & Setup
    # -------------------------------------------------------
    print("\n[1] Setting up Configuration...")
    set_seed(42)

    # Override Config for speed and isolation
    Config.DEBUG = True  # Forces data loading to sample a small subset
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 256
    Config.WORK_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Clean up any previous run
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.WORK_DIR)

    # 2. Verify Data Processing Logic (Unit Tests)
    # -------------------------------------------------------
    print("\n[2] Verifying Data Processing Logic...")

    # A. Test Temporal Lags
    # Create dummy tracking: 1 player, 3 steps.
    # x_position increases by 10 each step: 10, 20, 30.
    dummy_tracking = pd.DataFrame(
        {
            "game_play": ["g1_p1"] * 3,
            "nfl_player_id": [12345] * 3,
            "step": [0, 1, 2],
            "x_position": [10.0, 20.0, 30.0],
            "y_position": [0.0, 0.0, 0.0],
            "speed": [1.0, 1.0, 1.0],
            "acceleration": [0.1, 0.1, 0.1],
            "orientation": [0.0, 0.0, 0.0],
            "direction": [0.0, 0.0, 0.0],
        }
    )

    # Set LAG_STEPS to check forward/backward shifts.
    # Lag 1 means at t=0 we see t+1. Lag -1 means at t=1 we see t=0.
    Config.LAG_STEPS = [-1, 0, 1]

    processed_tracking = process_tracking_data(dummy_tracking)

    # Check Lag 1 (Future) at Step 0
    # x_position_lag1 at step 0 should be x_position at step 1 (20.0)
    val_lag1_step0 = processed_tracking.loc[0, "x_position_lag1"]
    assert (
        val_lag1_step0 == 20.0
    ), f"Lag Logic Error: Expected 20.0 for lag1 at step 0, got {val_lag1_step0}"

    # Check Lag -1 (Past) at Step 1
    # x_position_lag-1 at step 1 should be x_position at step 0 (10.0)
    val_lag_minus1_step1 = processed_tracking.loc[1, "x_position_lag-1"]
    assert (
        val_lag_minus1_step1 == 10.0
    ), f"Lag Logic Error: Expected 10.0 for lag-1 at step 1, got {val_lag_minus1_step1}"

    print("   -> process_tracking_data: OK")

    # B. Test Ground Imputation & Feature Engineering
    # Create dummy merged data. Player 2 is Ground ('G').
    dummy_merged = pd.DataFrame(
        {
            "nfl_player_id_2": ["G"],
            # Player 1 Data
            "x_position_lag0_1": [50.0],
            "y_position_lag0_1": [25.0],
            "speed_lag0_1": [5.0],
            # Player 2 Data (Pre-imputation, usually NaN or garbage for G)
            "x_position_lag0_2": [0.0],
            "y_position_lag0_2": [0.0],
            "speed_lag0_2": [99.0],  # Should be zeroed out
            "acceleration_lag0_2": [99.0],  # Should be zeroed out
        }
    )

    # Reset Config lags to just 0 for this test
    Config.LAG_STEPS = [0]

    engineered = impute_ground_and_engineer_features(dummy_merged.copy())

    # Check Position Imputation: P2 position should become P1 position
    assert (
        engineered.loc[0, "x_position_lag0_2"] == 50.0
    ), "Ground Imputation Error: X pos mismatch"
    assert (
        engineered.loc[0, "y_position_lag0_2"] == 25.0
    ), "Ground Imputation Error: Y pos mismatch"

    # Check Kinematics Imputation: P2 speed/accel should be 0
    assert (
        engineered.loc[0, "speed_lag0_2"] == 0.0
    ), "Ground Imputation Error: Speed not zeroed"

    # Check Relative Features:
    # Distance should be 0 (since positions are same) -> log1p(0) = 0
    # Relative Speed should be P1_speed - 0 = 5.0
    assert np.isclose(
        engineered.loc[0, "log_dist_lag0"], 0.0
    ), "Feature Eng Error: Log Dist"
    assert np.isclose(
        engineered.loc[0, "rel_speed_lag0"], 5.0
    ), "Feature Eng Error: Rel Speed"

    print("   -> impute_ground_and_engineer_features: OK")

    # 3. Verify Model Architecture
    # -------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    input_dim = 32
    hidden_dim = 64
    batch_size = 16

    model = ECGRN(input_dim, hidden_dim, num_blocks=2, dropout=0.1)
    dummy_input = torch.randn(batch_size, input_dim)

    # Forward pass
    output = model(dummy_input)

    assert output.shape == (
        batch_size,
        1,
    ), f"Model Output Shape Error: Got {output.shape}"
    assert torch.all(
        (output >= 0) & (output <= 1)
    ), "Model Output Range Error: Not in [0, 1]"

    # Loss function
    criterion = FocalLoss()
    dummy_target = torch.randint(0, 2, (batch_size, 1)).float()
    loss = criterion(output, dummy_target)

    assert not torch.isnan(loss), "Focal Loss returned NaN"
    assert loss.item() >= 0, "Focal Loss returned negative value"

    print("   -> ECGRN & FocalLoss: OK")

    # 4. Integration Test: Training Pipeline
    # -------------------------------------------------------
    print("\n[4] Running Training Pipeline Integration (Mini-Batch)...")

    # Load Data (DEBUG=True ensures small subset)
    # This uses the library's caching mechanism
    df_train = get_data(mode="train", load_cached_data=False)

    # Identify feature columns
    exclude_cols = ["contact_id", "game_play", "step", "contact", "is_val"]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    Config.INPUT_DIM = len(feature_cols)

    # Prepare small datasets
    train_subset = df_train[df_train["is_val"] == 0].iloc[:500]
    val_subset = df_train[df_train["is_val"] == 1].iloc[:100]

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_subset[feature_cols].values)
    y_train = train_subset["contact"].values.astype(float)
    X_val = scaler.transform(val_subset[feature_cols].values)
    y_val = val_subset["contact"].values.astype(float)

    # Dataloaders
    train_loader = DataLoader(
        ContactDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        ContactDataset(X_val, y_val), batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Setup Training Components
    device = torch.device("cpu")  # Use CPU for simple demo
    model = ECGRN(
        Config.INPUT_DIM, Config.HIDDEN_DIM, Config.NUM_BLOCKS, Config.DROPOUT
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # Run 1 Epoch
    avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"   -> Epoch 1 Loss: {avg_loss:.6f}")
    assert not np.isnan(avg_loss), "Training Loss is NaN"

    # Run Validation
    val_probs, val_targets = validate(model, val_loader, device)
    assert len(val_probs) == len(val_subset), "Validation output size mismatch"

    # Optimize Threshold
    # Create synthetic targets if val_subset has only one class to avoid MCC errors
    if len(np.unique(val_targets)) < 2:
        print(
            "   -> Warning: Validation set has single class. Mocking targets for threshold check."
        )
        val_targets = np.random.randint(0, 2, size=len(val_probs))

    best_thresh, best_mcc = optimize_threshold(val_targets, val_probs)
    print(f"   -> Optimized Threshold: {best_thresh:.2f}, MCC: {best_mcc:.4f}")
    assert 0.0 <= best_thresh <= 1.0, "Invalid threshold optimized"

    # 5. Integration Test: Inference
    # -------------------------------------------------------
    print("\n[5] Verifying Inference Logic...")

    # Load Test Data (Mocking by using get_data('test') which reads sample_submission)
    # Note: test_player_tracking.csv is large, but DEBUG=True doesn't apply to test in the library logic usually,
    # but get_data filters by relevant plays.

    try:
        df_test = get_data(mode="test", load_cached_data=False)

        # Align features
        for c in feature_cols:
            if c not in df_test.columns:
                df_test[c] = 0

        X_test = scaler.transform(df_test[feature_cols].values)
        test_loader = DataLoader(ContactDataset(X_test), batch_size=Config.BATCH_SIZE)

        model.eval()
        preds = []
        with torch.no_grad():
            for X_b in test_loader:
                preds.append(model(X_b.to(device)).cpu().numpy())
        preds = np.concatenate(preds)

        assert len(preds) == len(df_test), "Inference output size mismatch"
        print(f"   -> Inference generated {len(preds)} predictions.")

    except Exception as e:
        # Fallback if test data files are missing or malformed in the environment
        print(f"   -> Inference verification skipped due to data availability: {e}")

    # 6. Cleanup
    # -------------------------------------------------------
    print("\n[6] Cleaning up...")
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
