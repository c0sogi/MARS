import os
import torch
import pandas as pd
import numpy as np
import sys

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.features import engineer_features
from library.dataset import prepare_datasets
from library.model import CuratedIdentityNet
from library.loss import MaskedL1Loss
from library.train import run_training
from library.inference import predict


def main():
    print("=== Ventilator Pressure Prediction: Library Usage Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration for Demo Mode...")

    # Override Config defaults for a fast demonstration
    Config.DEBUG = True  # Enables data subsampling (critical for speed)
    Config.EXP_NAME = "demo_run"  # Separate directory for demo outputs
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo script

    # Initialize directories and random seeds
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Feature Engineering...")

    # We explicitly set load_cached_data=False to force processing logic execution
    # In DEBUG mode, this processes a small subset of breaths (e.g., 100 breaths)
    df_train = engineer_features("train", load_cached_data=False)

    # Validation
    print(f"Processed Train DataFrame Shape: {df_train.shape}")
    assert not df_train.empty, "DataFrame should not be empty"

    # Verify expected columns from feature engineering
    expected_cols = ["u_in_lag1", "u_in_diff1", "volume", "u_in_R", "vol_C"]
    for col in expected_cols:
        assert col in df_train.columns, f"Missing engineered feature: {col}"

    print("Feature engineering validation passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Data Loading...")

    # prepare_datasets handles tensor conversion, caching, and DataLoader creation
    train_loader, val_loader, test_loader, test_ids = prepare_datasets(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,  # Use the parquet cache generated in step 2
    )

    # Inspect a single batch
    batch = next(iter(train_loader))
    x, static, u_out, y = batch["x"], batch["static"], batch["u_out"], batch["y"]

    print(f"Input (x) shape: {x.shape}")  # Expected: (Batch, Seq_Len, Input_Dim)
    print(f"Static features shape: {static.shape}")  # Expected: (Batch, 2)
    print(f"Target (y) shape: {y.shape}")  # Expected: (Batch, Seq_Len)

    # Assertions
    assert x.shape[0] == Config.BATCH_SIZE
    assert x.shape[1] == Config.SEQ_LEN
    assert static.shape[1] == 2  # R and C
    assert y.shape == (Config.BATCH_SIZE, Config.SEQ_LEN)
    assert u_out.shape == y.shape

    print("DataLoader validation passed.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Model Forward Pass...")

    device = torch.device("cpu")  # Use CPU for assertion simplicity
    model = CuratedIdentityNet().to(device)

    # Move batch to device
    x_dev = x.to(device)
    static_dev = static.to(device)

    # Forward pass
    # Model returns (final_prediction, aux_prediction)
    pred_final, pred_aux = model(x_dev, static_dev)

    print(f"Prediction shape: {pred_final.shape}")

    # Assertions
    # Output should be (Batch, Seq_Len, 1)
    assert pred_final.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, 1)
    if pred_aux is not None:
        assert pred_aux.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, 1)

    print("Model forward pass validation passed.")

    # -------------------------------------------------------------------------
    # 5. Loss Calculation
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Loss Calculation...")

    criterion = MaskedL1Loss()

    # Prepare inputs for loss (squeeze last dim of pred to match y)
    loss_val = criterion(pred_final.squeeze(-1), y.to(device), u_out.to(device))

    print(f"Calculated Loss: {loss_val.item():.4f}")

    # Assertions
    assert isinstance(loss_val, torch.Tensor)
    assert loss_val.item() >= 0
    assert not torch.isnan(loss_val)

    print("Loss calculation validation passed.")

    # -------------------------------------------------------------------------
    # 6. Full Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[6] Executing Training Pipeline...")

    # run_training encapsulates the loop, validation, and saving logic
    # It will use the Config settings we modified (1 epoch, debug mode)
    run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Verify model artifact
    assert os.path.exists(Config.MODEL_PATH), f"Model not found at {Config.MODEL_PATH}"
    print("Training pipeline completed successfully.")

    # -------------------------------------------------------------------------
    # 7. Inference
    # -------------------------------------------------------------------------
    print("\n[7] Executing Inference...")

    inference_output_path = os.path.join(Config.WORKING_DIR, "inference_submission.csv")

    # Run prediction using the trained model
    predict(
        model_path=Config.MODEL_PATH,
        output_path=inference_output_path,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        device="cpu",  # Ensure compatibility
        load_cached_data=True,
    )

    # Verify submission file
    assert os.path.exists(inference_output_path), "Inference output file missing"

    sub_df = pd.read_csv(inference_output_path)
    print(f"Submission DataFrame Shape: {sub_df.shape}")
    print(f"Submission Head:\n{sub_df.head()}")

    # Assertions
    assert "id" in sub_df.columns
    assert "pressure" in sub_df.columns
    # In debug mode, we expect rows equal to (num_debug_breaths * seq_len)
    # The debug subset size is hardcoded to 100 breaths in dataset.py, seq_len is 80
    # So we expect roughly 8000 rows (or fewer if test set is smaller, but here it's subsampled)
    assert len(sub_df) > 0

    print("Inference validation passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
