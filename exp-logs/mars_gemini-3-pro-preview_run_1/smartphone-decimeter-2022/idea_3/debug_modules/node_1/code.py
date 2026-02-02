import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, compute_metric
from library.data_loader import get_dataloaders
from library.model import GnssDeepSetTCN
from library.trainer import Trainer


def main():
    print("=== Google Smartphone Decimeter Challenge: Library Demo ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # ------------------------------------------------------------------------
    print("\n[1] Configuring for fast demonstration...")

    # Modify Config attributes to run a small-scale test
    Config.DEBUG = True
    Config.DEBUG_SIZE = 200  # Use only 200 samples for training/val/test
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debugging
    Config.CACHE_DATA = True  # Enable caching to ./working directory

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Debug Size: {Config.DEBUG_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # ------------------------------------------------------------------------
    # 2. Data Loading Verification
    # ------------------------------------------------------------------------
    print("\n[2] Initializing Data Loaders...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders()
    except Exception as e:
        print(f"Error initializing dataloaders: {e}")
        # If metadata files don't exist (e.g. in a fresh environment without the generation step),
        # we can't proceed. Assuming they exist as per problem description.
        raise e

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Verify structure of a single batch
    print("Verifying batch structure...")
    try:
        batch = next(iter(train_loader))
        sat_feat, glob_feat, mask, target, wls_lla = batch

        print(f"  Satellite Features: {sat_feat.shape} (Batch, MaxSat, Feats)")
        print(f"  Global Features:    {glob_feat.shape} (Batch, Feats)")
        print(f"  Masks:              {mask.shape} (Batch, MaxSat)")
        print(f"  Targets:            {target.shape} (Batch, 2)")
        print(f"  WLS LLA:            {wls_lla.shape} (Batch, 3)")

        # Assertions to ensure data integrity
        assert sat_feat.ndim == 3, "Satellite features should be 3D"
        assert glob_feat.ndim == 2, "Global features should be 2D"
        assert target.shape[1] == 2, "Target should have 2 columns (Lat, Lon residuals)"
        assert not torch.isnan(
            sat_feat
        ).all(), "Satellite features should not be all NaN"

        print("Data Loader verification passed.")

    except StopIteration:
        print("Train loader is empty! Please check if metadata contains valid paths.")
        return

    # ------------------------------------------------------------------------
    # 3. Model Instantiation and Forward Pass
    # ------------------------------------------------------------------------
    print("\n[3] Testing Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = GnssDeepSetTCN().to(device)

    # Prepare inputs for the model (add sequence dimension as model expects TCN input)
    # The DataLoader provides (B, ...) but model expects (B, SeqLen, ...)
    # Here SeqLen = 1 for independent epochs
    sat_input = sat_feat.to(device).unsqueeze(1)
    glob_input = glob_feat.to(device).unsqueeze(1)
    mask_input = mask.to(device).unsqueeze(1)

    print(f"  Input shapes: {sat_input.shape}, {glob_input.shape}")

    try:
        with torch.no_grad():
            output = model(sat_input, glob_input, mask_input)
            # Remove sequence dimension for output verification
            output = output.squeeze(1)

        print(f"  Output shape: {output.shape}")

        assert output.shape == (sat_feat.size(0), 2), "Model output shape mismatch"
        assert not torch.isnan(output).any(), "Model produced NaN outputs"

        print("Model forward pass verification passed.")
    except Exception as e:
        print(f"Model forward pass failed: {e}")
        raise e

    # ------------------------------------------------------------------------
    # 4. Training Loop Simulation
    # ------------------------------------------------------------------------
    print("\n[4] Running Training Loop...")
    trainer = Trainer(train_loader, val_loader, test_loader)

    # Run fit (1 epoch as configured)
    trainer.fit()

    # Verify model checkpoint creation
    if os.path.exists(trainer.best_model_path):
        print(f"Model successfully saved to {trainer.best_model_path}")
    else:
        # If validation score didn't improve (unlikely with inf init), might not save.
        # But Trainer logic saves on first epoch if score < inf.
        print("Warning: Model weights file not found.")

    # ------------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------------
    print("\n[5] Evaluating on Validation Set...")
    val_score, val_loss = trainer.evaluate(val_loader)
    print(f"  Validation Loss (MAE): {val_loss:.6f}")
    print(f"  Validation Score (Mean 50/95): {val_score:.6f}")

    assert val_score > 0, "Validation score should be positive"

    # ------------------------------------------------------------------------
    # 6. Inference / Prediction
    # ------------------------------------------------------------------------
    print("\n[6] Generating Submission...")
    trainer.predict()

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"  Rows in submission: {len(df_sub)}")
        print(f"  Columns: {list(df_sub.columns)}")

        expected_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        assert all(
            col in df_sub.columns for col in expected_cols
        ), "Missing columns in submission"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    # ------------------------------------------------------------------------
    # 7. Metric Calculation Utility Verification
    # ------------------------------------------------------------------------
    print("\n[7] Verifying Metric Utility...")
    # Create synthetic ground truth and predictions
    df_gt = pd.DataFrame(
        {
            "tripId": ["trip_A", "trip_A", "trip_B"],
            "UnixTimeMillis": [1000, 2000, 1000],
            "LatitudeDegrees": [37.0, 37.0001, 40.0],
            "LongitudeDegrees": [-122.0, -122.0, -74.0],
        }
    )

    # Predictions with known errors
    # Point 1: Exact match (Error 0)
    # Point 2: Offset by ~11m (0.0001 deg lat is approx 11.1m)
    # Point 3: Exact match (Error 0)
    df_pred = pd.DataFrame(
        {
            "tripId": ["trip_A", "trip_A", "trip_B"],
            "UnixTimeMillis": [1000, 2000, 1000],
            "LatitudeDegrees": [37.0, 37.0, 40.0],
            "LongitudeDegrees": [-122.0, -122.0, -74.0],
        }
    )

    score = compute_metric(df_pred, df_gt)
    print(f"  Computed Score on synthetic data: {score:.4f}")

    # Expected logic:
    # Trip A errors: [0, ~11.1m]. 50th=5.55, 95th=~10.5. Mean(50,95) ~ 8.0
    # Trip B errors: [0]. 50th=0, 95th=0. Mean(50,95) = 0
    # Average across phones: (~8.0 + 0) / 2 = ~4.0

    assert score > 0, "Score should be positive given the error introduced."
    assert score < 100, "Score should be reasonable for small degree offsets."
    print("Metric utility verification passed.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
