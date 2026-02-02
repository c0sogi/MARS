import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import library components
from library.config import Config
from library.data_processing import get_data_loaders
from library.model import VentilatorNet
from library.train_eval import MaskedL1Loss, train_one_epoch, evaluate, predict


def run_demo():
    print("=== Starting Ventilator Pressure Prediction Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configure for Fast Demonstration
    # ---------------------------------------------------------
    print("[1] Configuring experiment settings for speed...")

    # Override Config defaults for a lightweight run
    Config.EXPERIMENT_NAME = "demo_run"
    Config.DEBUG = True  # Uses tiny subset (100 train breaths, 50 val/test)
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16

    # Reduce Model Complexity for Demo
    Config.D_MODEL = 64
    Config.LSTM_HIDDEN = 32  # Must be D_MODEL / 2 for Bi-LSTM identity residual
    Config.NUM_LAYERS = 2

    # Re-run setup to create directories for 'demo_run' and set seeds
    Config.setup()

    print(f"    Experiment: {Config.EXPERIMENT_NAME}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Processing and Loading
    # ---------------------------------------------------------
    print("\n[2] Processing data and creating DataLoaders...")

    # Force processing from scratch (load_cached_data=False) to verify feature engineering
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=False)

    # Verify Data Shapes
    # Batch structure: x (Batch, 80, 13), u_out (Batch, 80), y (Batch, 80)
    sample_x, sample_uout, sample_y = next(iter(train_loader))

    print(f"    Train Batch X shape: {sample_x.shape}")
    print(f"    Train Batch y shape: {sample_y.shape}")

    assert sample_x.shape == (Config.BATCH_SIZE, 80, 13), "Incorrect input shape"
    assert sample_uout.shape == (Config.BATCH_SIZE, 80), "Incorrect u_out shape"
    assert sample_y.shape == (Config.BATCH_SIZE, 80), "Incorrect target shape"

    print("    Assertion Passed: Data shapes are correct.")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[3] Initializing VentilatorNet...")

    device = torch.device(Config.DEVICE)
    model = VentilatorNet().to(device)

    # Verify Model Output
    sample_x = sample_x.to(device)
    sample_uout = sample_uout.to(device)

    # Forward pass (training mode returns final_pred, aux_pred)
    model.train()
    pred, aux_pred = model(sample_x, sample_uout)

    print(f"    Prediction shape: {pred.shape}")
    print(f"    Aux Prediction shape: {aux_pred.shape}")

    assert pred.shape == (Config.BATCH_SIZE, 80), "Prediction shape mismatch"
    assert aux_pred.shape == (Config.BATCH_SIZE, 80), "Aux prediction shape mismatch"

    print("    Assertion Passed: Model forward pass successful.")

    # ---------------------------------------------------------
    # 4. Training Loop Simulation
    # ---------------------------------------------------------
    print("\n[4] Simulating Training Loop...")

    criterion = MaskedL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Simple scheduler for demo
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    # Run Training
    for epoch in range(Config.EPOCHS):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_mae = evaluate(model, val_loader, criterion, device)
        print(
            f"    Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_loss:.4f} | Val MAE: {val_mae:.4f}"
        )

        # Verify Loss is valid
        assert not np.isnan(avg_loss), "Training loss is NaN"
        assert val_mae >= 0, "Validation MAE is negative"

    # Save the demo model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"    Model saved to {Config.MODEL_SAVE_PATH}")

    # ---------------------------------------------------------
    # 5. Inference and Submission Generation
    # ---------------------------------------------------------
    print("\n[5] Running Inference on Test Set...")

    # Predict
    predictions = predict(model, test_loader, device)

    # In Debug mode, test set is truncated to 50 breaths * 80 steps = 4000 steps
    expected_len = 50 * 80
    print(f"    Generated {len(predictions)} predictions. Expected: {expected_len}")

    assert (
        len(predictions) == expected_len
    ), f"Prediction count mismatch. Got {len(predictions)}, expected {expected_len}"

    # Load IDs for submission
    # In a real run, we load from cache. Here we reconstruct the path logic or load directly for verification.
    # Since we are in debug mode, we need the IDs corresponding to the truncated test set.
    # The get_data_loaders function saved 'test_ids' to cache.

    # Reconstruct cache filename hash logic to find the file
    import hashlib

    feature_version = "v1_physics_robust_uniform"
    cache_hash = hashlib.md5(
        f"{feature_version}_{Config.DEBUG}_{Config.EXPERIMENT_NAME}".encode()
    ).hexdigest()
    test_ids_path = os.path.join(Config.CACHE_DIR, f"test_ids_{cache_hash}.npy")

    if os.path.exists(test_ids_path):
        test_ids = np.load(test_ids_path).flatten()
        print(f"    Loaded {len(test_ids)} test IDs from cache.")

        assert len(test_ids) == len(predictions), "ID and Prediction length mismatch"

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "pressure": predictions})

        # Save
        submission_path = os.path.join(Config.WORKING_DIR, "inference_submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"    Submission saved to {submission_path}")

        # Verify file content
        saved_df = pd.read_csv(submission_path)
        print(f"    Verification: Saved CSV has {len(saved_df)} rows.")
        assert saved_df.shape[1] == 2, "Submission should have 2 columns"
        assert (
            "id" in saved_df.columns and "pressure" in saved_df.columns
        ), "Missing columns in submission"
    else:
        raise FileNotFoundError(
            "Test IDs cache file not found, cannot generate submission."
        )

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
