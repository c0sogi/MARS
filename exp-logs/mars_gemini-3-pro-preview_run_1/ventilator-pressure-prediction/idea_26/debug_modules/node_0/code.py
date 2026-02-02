import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device, compute_metric
from library.data import prepare_data, get_feature_columns, engineer_features
from library.model import WideProjectedNet
from library.train import (
    Trainer,
    MaskedHybridLoss,
    get_feature_names,
    generate_submission,
)


def main():
    print("=== Ventilator Pressure Prediction Demo Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    print("\n[Step 1] Setting up Configuration...")

    # Initialize default directories and seeds
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # OVERRIDE Config for Speed/Demo purposes
    print("Overriding configuration for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 400  # Use a very small subset of breaths
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory is clean for the demo run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Preparation and Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Preparing Data...")

    # prepare_data handles loading, feature engineering, scaling, and reshaping
    train_loader, val_loader, test_loader, test_ids = prepare_data()

    # Verify Data Loaders
    print("Verifying Train Loader...")
    try:
        X_batch, u_out_batch, y_batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # Assert Shapes
    # Expected: (Batch, Seq_Len, Features)
    assert X_batch.dim() == 3, f"Expected 3D input, got {X_batch.dim()}"
    assert (
        X_batch.size(0) == Config.BATCH_SIZE
    ), f"Batch size mismatch: {X_batch.size(0)} vs {Config.BATCH_SIZE}"
    assert (
        X_batch.size(1) == Config.SEQ_LEN
    ), f"Seq len mismatch: {X_batch.size(1)} vs {Config.SEQ_LEN}"

    # Expected Target: (Batch, Seq_Len)
    assert y_batch.dim() == 2, f"Expected 2D target, got {y_batch.dim()}"
    assert y_batch.size() == (Config.BATCH_SIZE, Config.SEQ_LEN)

    # Expected u_out: (Batch, Seq_Len)
    assert u_out_batch.size() == (Config.BATCH_SIZE, Config.SEQ_LEN)

    print(f"  Batch X shape: {X_batch.shape}")
    print(f"  Batch y shape: {y_batch.shape}")
    print("Data Loader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Logic Check
    # -------------------------------------------------------------------------
    print("\n[Step 3] Initializing Model...")

    # We need feature names to configure the model's context awareness
    feature_names = get_feature_names()
    input_dim = len(feature_names)
    print(f"  Detected Input Dimension: {input_dim}")

    model = WideProjectedNet(input_dim=input_dim, feature_names=feature_names)
    model = model.to(device)

    # Verify Forward Pass
    print("Verifying Model Forward Pass...")
    X_batch = X_batch.to(device)

    with torch.no_grad():
        final_pred, aux_pred = model(X_batch)

    # Check Output Shapes
    # Output should be (Batch, Seq, 1)
    assert final_pred.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        1,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, 1)}, got {final_pred.shape}"

    if aux_pred is not None:
        assert aux_pred.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, 1)

    print("Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 4. Metric and Loss Logic Verification
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Metric and Loss Logic...")

    # Create synthetic data
    # Case: Error exists ONLY in expiratory phase (u_out=1).
    # Since metric only cares about inspiratory (u_out=0), MAE should be 0.

    synth_pred = torch.tensor([10.0, 10.0, 20.0, 20.0])  # 2 insp, 2 exp
    synth_true = torch.tensor([10.0, 10.0, 10.0, 10.0])  # True values
    synth_u_out = torch.tensor([0.0, 0.0, 1.0, 1.0])  # 0=Insp, 1=Exp

    # Metric Check
    mae = compute_metric(synth_pred, synth_true, synth_u_out)
    print(f"  Computed MAE (Expected 0.0): {mae}")
    assert np.isclose(mae, 0.0), f"Metric failed. Expected 0.0, got {mae}"

    # Loss Check
    # MaskedHybridLoss should also ignore the error in the expiratory phase
    criterion = MaskedHybridLoss()
    loss = criterion(synth_pred, None, synth_true, synth_u_out)
    print(f"  Computed Loss (Expected 0.0): {loss.item()}")
    assert np.isclose(loss.item(), 0.0), f"Loss failed. Expected 0.0, got {loss.item()}"

    print("Metric and Loss verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Training Loop (1 Epoch)...")

    trainer = Trainer(model, device, train_loader, val_loader)
    trainer.fit()

    # Verify Model Save
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file was not saved to {Config.MODEL_SAVE_PATH}")

    print(f"Model successfully saved to {Config.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[Step 6] Generating Submission...")

    # Load the saved model to ensure weight loading works
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, test_ids, device)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_FILE_PATH):
        raise FileNotFoundError("Submission file not found.")

    sub_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"  Submission shape: {sub_df.shape}")

    # Check columns
    assert list(sub_df.columns) == ["id", "pressure"], "Submission columns mismatch"

    # Check length matches test_ids
    assert len(sub_df) == len(
        test_ids
    ), f"Submission length {len(sub_df)} does not match test set size {len(test_ids)}"

    print("Submission verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
