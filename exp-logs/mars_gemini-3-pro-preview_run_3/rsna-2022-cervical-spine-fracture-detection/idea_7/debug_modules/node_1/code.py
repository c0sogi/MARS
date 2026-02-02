import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from unittest.mock import MagicMock

# --- 1. Environment Mocking ---
# dataset.py imports pydicom. If it's missing in the environment, we mock it
# to ensure the script demonstrates the pipeline logic without crashing.
try:
    import pydicom
except ImportError:
    pydicom = MagicMock()
    # Setup mock to return a valid numpy array for pixel_array so windowing works
    mock_dataset = MagicMock()
    mock_dataset.pixel_array = np.zeros((512, 512), dtype=np.int16)
    mock_dataset.RescaleIntercept = 0
    mock_dataset.RescaleSlope = 1
    mock_dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
    pydicom.dcmread.return_value = mock_dataset
    sys.modules["pydicom"] = pydicom

# --- 2. Library Imports ---
from library.config import Config
from library.utils import seed_everything, calculate_weighted_loss
from library.dataset import CervicalSpineDataset
from library.model import CervicalFractureModel
from library.engine import fit


def main():
    # --- 3. Setup & Configuration ---
    print(">>> Setting up configuration...")
    seed_everything(42)

    # Configure for a fast demonstration run
    # Reduce slices to 8 (from 64) to speed up volume construction
    # Reduce batch size to 2 and epochs to 1
    Config.setup(debug=True, epochs=1, batch_size=2, n_slices=8)

    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # --- 4. Dataset & Pipeline Verification ---
    print(">>> Verifying Dataset and Transforms...")
    # Initialize dataset
    train_ds = CervicalSpineDataset(phase="train")

    if len(train_ds) > 0:
        # Fetch one sample to check shapes
        volume, label, study_id = train_ds[0]

        # Expected Volume Shape: (N_Slices, Channels, H, W)
        # Channels = 3 (2.5D context), H=W=256 (Config.IMAGE_SIZE)
        expected_vol_shape = (Config.N_SLICES, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        expected_label_shape = (8,)  # patient_overall + C1..C7

        print(f"Volume Shape: {volume.shape}")
        print(f"Label Shape: {label.shape}")

        if volume.shape != expected_vol_shape:
            raise AssertionError(
                f"Volume shape mismatch. Expected {expected_vol_shape}, got {volume.shape}"
            )
        if label.shape != expected_label_shape:
            raise AssertionError(
                f"Label shape mismatch. Expected {expected_label_shape}, got {label.shape}"
            )
    else:
        print("Warning: Dataset is empty (check metadata). Skipping item verification.")

    # --- 5. Model Architecture Verification ---
    print(">>> Verifying Model Architecture...")
    # Initialize model (pretrained=False for speed/offline safety)
    model = CervicalFractureModel(n_classes=7, pretrained=False)
    model.to(Config.DEVICE)

    # Create dummy input: (Batch, N_Slices, Channels, H, W)
    dummy_input = torch.randn(
        2, Config.N_SLICES, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Model Output Shape: {logits.shape}")
    # Expected output: (Batch, 8)
    if logits.shape != (2, 8):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, 8), got {logits.shape}"
        )

    # --- 6. Training Loop Demonstration ---
    print(">>> Running Training Loop Demo...")

    # Use a tiny subset of data for the demo
    subset_indices = list(range(min(4, len(train_ds))))
    train_subset = Subset(train_ds, subset_indices)
    val_subset = Subset(train_ds, subset_indices)  # Reuse for validation demo

    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Run training for 1 epoch
    fit(model, train_loader, val_loader, Config.DEVICE, epochs=Config.EPOCHS)

    # Check if model was saved
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise AssertionError("Model checkpoint was not saved.")
    print("Training demo completed successfully.")

    # --- 7. Metric Verification ---
    print(">>> Verifying Metric Logic...")
    # Create synthetic ground truth and predictions
    y_true = pd.DataFrame(
        {
            "StudyInstanceUID": ["study_1"],
            "patient_overall": [1],
            "C1": [0],
            "C2": [0],
            "C3": [0],
            "C4": [1],
            "C5": [0],
            "C6": [0],
            "C7": [0],
        }
    )

    # Perfect prediction case
    y_pred_perfect = pd.DataFrame(
        {
            "StudyInstanceUID": ["study_1"],
            "patient_overall": [0.99],
            "C1": [0.01],
            "C2": [0.01],
            "C3": [0.01],
            "C4": [0.99],
            "C5": [0.01],
            "C6": [0.01],
            "C7": [0.01],
        }
    )

    loss = calculate_weighted_loss(y_true, y_pred_perfect)
    print(f"Calculated Loss (Perfect Preds): {loss:.6f}")

    # Loss should be very small
    if loss > 0.1:
        raise AssertionError(
            "Metric calculation seems incorrect; loss is too high for perfect predictions."
        )

    # --- 8. Inference & Submission Formatting ---
    print(">>> Simulating Inference and Submission...")

    # Simulate model output for a test study
    test_study_id = "1.2.826.0.1.3680043.10001"
    # Random probabilities for 8 classes
    probs = np.random.rand(8)

    # Map to submission format
    # Columns: row_id, fractured
    # row_id format: {StudyInstanceUID}_{TargetName}
    target_names = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    submission_rows = []
    for i, target in enumerate(target_names):
        row_id = f"{test_study_id}_{target}"
        prob = probs[i]
        submission_rows.append({"row_id": row_id, "fractured": prob})

    submission_df = pd.DataFrame(submission_rows)
    print("Sample Submission Rows:")
    print(submission_df.head(8))

    if len(submission_df) != 8:
        raise AssertionError("Submission generation failed to create 8 rows per study.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
