import os
import sys
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import MIPLinearDecayNet
from library.utils import LaplaceLogLikelihoodLoss
from library.engine import Trainer


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print("Initializing demonstration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config for a fast, minimal demonstration
    print("Configuring for speed (Debug Mode)...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 10  # Use only 10 samples
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.LOAD_CACHED_DATA = (
        False  # Force processing logic to run (or fallback to black images)
    )

    # Ensure working directory exists for model saving
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print("\n[Step 1] Testing Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch a single batch to verify shapes
    try:
        batch = next(iter(train_loader))
        images = batch["image"]
        tabular = batch["tabular"]
        base_fvc = batch["base_fvc"]
        weeks = batch["weeks"]
        true_fvc = batch["fvc_true"]

        print(f"  Batch Loaded.")
        print(
            f"  Image Shape: {images.shape} (Expected: [{Config.BATCH_SIZE}, 1, {Config.IMG_SIZE}, {Config.IMG_SIZE}])"
        )
        print(
            f"  Tabular Shape: {tabular.shape} (Expected: [{Config.BATCH_SIZE}, {Config.N_TABULAR_FEATURES}])"
        )

        # Assertions
        assert images.shape == (
            Config.BATCH_SIZE,
            1,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), "Incorrect Image Tensor shape"
        assert tabular.shape == (
            Config.BATCH_SIZE,
            Config.N_TABULAR_FEATURES,
        ), "Incorrect Tabular Tensor shape"
        assert (
            base_fvc.shape[0] == Config.BATCH_SIZE
        ), "Incorrect Batch Size for metadata"

    except Exception as e:
        print(f"  Data Loading Failed: {e}")
        raise e

    # ==========================================
    # 3. Model Demonstration
    # ==========================================
    print("\n[Step 2] Testing Model Initialization and Forward Pass...")
    device = Config.get_device()
    print(f"  Device: {device}")

    model = MIPLinearDecayNet().to(device)

    # Move batch to device
    images = images.to(device)
    tabular = tabular.to(device)

    # Forward pass
    slope, confidence = model(images, tabular)

    print(
        f"  Model Output Shapes -> Slope: {slope.shape}, Confidence: {confidence.shape}"
    )

    # Assertions
    assert slope.shape == (Config.BATCH_SIZE, 1), "Slope output shape mismatch"
    assert confidence.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Confidence output shape mismatch"
    # Confidence must be positive (Softplus used in model)
    assert (confidence > 0).all(), "Confidence values must be positive"

    # ==========================================
    # 4. Loss Function Demonstration
    # ==========================================
    print("\n[Step 3] Testing Loss Function...")
    criterion = LaplaceLogLikelihoodLoss()

    # Move targets to device
    base_fvc = base_fvc.to(device)
    weeks = weeks.to(device)
    true_fvc = true_fvc.to(device)

    # Calculate loss
    loss = criterion(slope, confidence, base_fvc, weeks, true_fvc)
    print(f"  Computed Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[Step 4] Testing Training Loop...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    trainer = Trainer(model, optimizer, device)

    # Run training for 1 epoch
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS, patience=1)

    # Verify model checkpoint
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"  Model successfully saved to {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created.")

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    print("\n[Step 5] Testing Inference and Submission Generation...")
    trainer.predict(test_loader)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"  Submission file created at {Config.SUBMISSION_PATH}")
        print(f"  Rows: {len(sub_df)}")
        print(f"  Columns: {list(sub_df.columns)}")

        expected_cols = ["Patient_Week", "FVC", "Confidence"]
        assert all(
            col in sub_df.columns for col in expected_cols
        ), "Missing columns in submission"
        assert len(sub_df) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n" + "=" * 40)
    print("SUCCESS: All pipeline components verified.")
    print("=" * 40)


if __name__ == "__main__":
    main()
