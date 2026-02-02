import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, RSNALoss
from library.dataset import RSNADataset
from library.model import FractureModel
from library.engine import fit_model, inference_and_submit


def main():
    print("=== Starting RSNA Fracture Detection Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # We modify the Config class attributes directly to optimize for a fast demo run.
    print("[Setup] Configuring parameters for demo execution...")

    seed_everything(42)

    # Enable debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Use only 16 samples for speed

    # Training hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2

    # Disable downloading pretrained weights to avoid network dependency/time
    Config.PRETRAINED = False

    # Ensure working directories are clean
    if os.path.exists(Config.MODEL_SAVE_PATH):
        os.remove(Config.MODEL_SAVE_PATH)

    # -------------------------------------------------------------------------
    # 2. Validate Dataset
    # -------------------------------------------------------------------------
    print("\n[Validation] Testing RSNADataset...")

    # Initialize dataset in debug mode
    train_ds = RSNADataset(subset="train", debug=True)

    # Verify dataset is not empty
    assert len(train_ds) > 0, "Dataset is empty."
    print(f"Dataset initialized with {len(train_ds)} samples.")

    # Fetch one item
    volume, target = train_ds[0]

    # Check shapes
    # Expected Volume: (Seq_Len=64, Channels=3, H=224, W=224)
    # Expected Target: (8,) -> [C1..C7, Patient_Overall]
    print(f"Sample Volume Shape: {volume.shape}")
    print(f"Sample Target Shape: {target.shape}")

    assert volume.shape == (64, 3, 224, 224), f"Incorrect volume shape: {volume.shape}"
    assert target.shape == (8,), f"Incorrect target shape: {target.shape}"
    assert volume.dtype == torch.float32, "Volume should be float32"

    # -------------------------------------------------------------------------
    # 3. Validate Model
    # -------------------------------------------------------------------------
    print("\n[Validation] Testing FractureModel...")

    # Initialize model (random weights)
    model = FractureModel(pretrained=False)
    model.eval()

    # Create a dummy batch: (Batch=2, Seq=64, C=3, H=224, W=224)
    dummy_input = torch.randn(2, 64, 3, 224, 224)

    # Run forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Expected Output: (Batch=2, Classes=8)
    assert output.shape == (2, 8), f"Incorrect output shape: {output.shape}"

    # -------------------------------------------------------------------------
    # 4. Validate Loss Function
    # -------------------------------------------------------------------------
    print("\n[Validation] Testing RSNALoss...")

    criterion = RSNALoss()

    # Dummy logits and targets
    dummy_logits = torch.randn(2, 8)
    dummy_targets = torch.randint(0, 2, (2, 8)).float()

    loss = criterion(dummy_logits, dummy_targets)

    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss must be positive"

    # -------------------------------------------------------------------------
    # 5. Integration Test: Training Loop
    # -------------------------------------------------------------------------
    print("\n[Integration] Running Training Loop (fit_model)...")

    # This runs the full training pipeline for 1 epoch on the debug subset
    fit_model()

    # Verify model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print("Training completed. Best model saved.")

    # -------------------------------------------------------------------------
    # 6. Integration Test: Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[Integration] Running Inference (inference_and_submit)...")

    # This loads the saved model and generates submission.csv
    inference_and_submit()

    # Verify submission file exists
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Validate submission content format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    expected_cols = ["row_id", "fractured"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Check if we have predictions for the test set
    # Note: In debug mode, the test set might be filtered or full depending on implementation.
    # We just verify the file is valid and non-empty.
    assert len(sub_df) > 0, "Submission file is empty."

    print("Inference completed. Submission file verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
