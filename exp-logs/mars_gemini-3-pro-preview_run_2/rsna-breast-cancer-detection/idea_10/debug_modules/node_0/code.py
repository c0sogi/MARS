import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import SMTSINModel
from library.train import run_training


def main():
    print("Starting End-to-End Demonstration...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    # We modify the Config class attributes directly to run a fast demo.
    print("Configuring parameters for fast execution...")
    Config.DEBUG = True  # Use a small subset of data (100 train, 50 val/test)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Reduced batch size for safety/speed
    Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead on small data

    # Ensure the working directory exists (though Config.create_dirs handles it)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n[Validation] Checking Data Pipeline...")
    # Load dataloaders in debug mode
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch a single batch from the training loader
    try:
        images, metadata, targets = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader returned no data.")

    # Verify Image Tensor
    # Expected: (Batch_Size, Channels=3, Height=640, Width=640)
    print(f"Batch Images Shape: {images.shape}")
    assert images.dim() == 4, "Images tensor must be 4-dimensional."
    assert images.shape[1] == 3, "Images must have 3 channels (Original + 2 CLAHE)."
    assert images.shape[2] == Config.IMG_HEIGHT, f"Height must be {Config.IMG_HEIGHT}."
    assert images.shape[3] == Config.IMG_WIDTH, f"Width must be {Config.IMG_WIDTH}."

    # Verify Metadata Tensor
    # Expected: (Batch_Size, 4) -> [Age, Implant, View_Idx, Machine_Idx]
    print(f"Batch Metadata Shape: {metadata.shape}")
    assert metadata.dim() == 2, "Metadata tensor must be 2-dimensional."
    assert metadata.shape[1] == 4, "Metadata must have 4 features."

    # Verify Targets Dictionary
    print(f"Target Keys: {list(targets.keys())}")
    assert "cancer" in targets, "Missing 'cancer' target."
    assert "birads" in targets, "Missing 'birads' target."
    assert "density" in targets, "Missing 'density' target."

    print("Data Pipeline check passed.")

    # ==========================================
    # 3. Model Logic Verification
    # ==========================================
    print("\n[Validation] Checking Model Logic...")
    model = SMTSINModel().to(device)

    # Move batch to device
    images = images.to(device)
    metadata = metadata.to(device)

    # Perform Forward Pass
    with torch.no_grad():
        outputs = model(images, metadata)

    # Verify Outputs
    # Cancer: (B, 1) logits
    # BIRADS: (B, 1) regression
    # Density: (B, 4) logits
    B = images.shape[0]

    assert "cancer" in outputs
    assert outputs["cancer"].shape == (
        B,
        1,
    ), f"Cancer output shape mismatch: {outputs['cancer'].shape}"

    assert "birads" in outputs
    assert outputs["birads"].shape == (
        B,
        1,
    ), f"BIRADS output shape mismatch: {outputs['birads'].shape}"

    assert "density" in outputs
    assert outputs["density"].shape == (
        B,
        4,
    ), f"Density output shape mismatch: {outputs['density'].shape}"

    print("Model Logic check passed.")

    # ==========================================
    # 4. End-to-End Training & Inference
    # ==========================================
    print("\n[Execution] Running Training and Inference Pipeline...")
    # This function handles the training loop, validation, saving best model, and generating submission
    run_training(debug=True)

    # ==========================================
    # 5. Submission Verification
    # ==========================================
    print("\n[Validation] Verifying Submission File...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Rows: {len(submission_df)}")
    print(submission_df.head(3))

    # Check Columns
    required_cols = ["prediction_id", "cancer"]
    for col in required_cols:
        if col not in submission_df.columns:
            raise AssertionError(f"Submission missing required column: {col}")

    # Check Probability Range
    probs = submission_df["cancer"]
    if not ((probs >= 0).all() and (probs <= 1).all()):
        raise AssertionError(
            "Submission contains invalid probabilities (outside 0-1 range)."
        )

    # Check Prediction ID format (Basic check)
    # prediction_id should be strings
    if not pd.api.types.is_string_dtype(submission_df["prediction_id"]):
        raise AssertionError("prediction_id column must be of string type.")

    print("Submission verification passed.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
