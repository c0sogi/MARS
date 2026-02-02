import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_log_loss
from library.dataset import get_dataloaders
from library.model import get_model, setup_phase
from library.trainer import Trainer


def main():
    print("=== Starting Dog Breed Classification Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    # We modify the Config class directly to run a fast demonstration
    # on a tiny subset of data without downloading heavy weights.
    print("[1] Configuring environment for rapid demonstration...")

    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 images per split
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debug
    Config.PHASE1_EPOCHS = 1  # Run only 1 epoch for Phase 1
    Config.PHASE2_EPOCHS = 1  # Run only 1 epoch for Phase 2
    Config.PRETRAINED = False  # Skip downloading ImageNet weights for speed

    # Set a custom working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_TRAIN_PATH = os.path.join(Config.WORKING_DIR, "train.parquet")
    Config.CACHE_VAL_PATH = os.path.join(Config.WORKING_DIR, "val.parquet")
    Config.CACHE_TEST_PATH = os.path.join(Config.WORKING_DIR, "test.parquet")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean up previous demo runs
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    print("    Configuration updated: DEBUG=True, Epochs=1, Subset=20")

    # -------------------------------------------------------------------------
    # 2. Utility Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Seed
    set_seed(42)

    # Test Log Loss
    # Scenario: 3 samples, 3 classes.
    # Sample 0: True=0, Pred=[0.8, 0.1, 0.1] (Good)
    # Sample 1: True=1, Pred=[0.1, 0.8, 0.1] (Good)
    # Sample 2: True=2, Pred=[0.1, 0.1, 0.8] (Good)
    y_true = np.array([0, 1, 2])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    loss = calculate_log_loss(y_true, y_pred)
    print(f"    Calculated Log Loss: {loss:.4f}")

    # Assert loss is reasonably low for good predictions
    assert loss < 0.5, f"Log loss {loss} is too high for accurate predictions."
    print("    Log Loss calculation verified.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Pipeline...")

    # Load DataLoaders (force reload to ignore any existing cache)
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=False
    )

    # Verify Class Count
    print(f"    Number of classes detected: {len(classes)}")
    assert len(classes) == 120, f"Expected 120 classes, found {len(classes)}"

    # Verify Batch Shapes
    images, labels = next(iter(train_loader))
    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.CROP_SIZE,
        Config.CROP_SIZE,
    ), "Incorrect image batch shape."
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape."

    print("    DataLoaders verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture & Phase Setup Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model and Phase Setup...")

    # Instantiate Model
    model = get_model(num_classes=len(classes), pretrained=Config.PRETRAINED)

    # Check Output Shape
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (
        2,
        120,
    ), f"Expected output shape (2, 120), got {output.shape}"
    print("    Model output shape verified.")

    # Test Phase 1 Setup (Head Adaptation)
    setup_phase(model, "phase1")
    # Backbone (e.g., features[0]) should be frozen
    assert (
        model.features[0][0].weight.requires_grad is False
    ), "Phase 1 Error: Backbone should be frozen."
    # Classifier head should be trainable
    assert (
        model.classifier[2].weight.requires_grad is True
    ), "Phase 1 Error: Classifier head should be trainable."
    print("    Phase 1 freezing logic verified.")

    # Test Phase 2 Setup (Fine-Tuning)
    setup_phase(model, "phase2")
    # Early backbone (Stage 0) should still be frozen
    assert (
        model.features[0][0].weight.requires_grad is False
    ), "Phase 2 Error: Early backbone should remain frozen."
    # Stage 4 (index 7) should be trainable
    # Accessing the first block of Stage 4
    assert (
        list(model.features[7].parameters())[0].requires_grad is True
    ), "Phase 2 Error: Stage 4 should be trainable."
    print("    Phase 2 freezing logic verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Trainer.fit)...")

    trainer = Trainer()
    # Run the fit process (Train -> Validate -> Save -> Predict)
    trainer.fit(load_cached_data=True)

    print("    Training loop completed.")

    # -------------------------------------------------------------------------
    # 6. Submission Validation
    # -------------------------------------------------------------------------
    print("\n[6] Validating Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Shape: {sub_df.shape}")

    # Verify Row Count (should match DEBUG_SUBSET_SIZE)
    assert (
        len(sub_df) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} rows, got {len(sub_df)}"

    # Verify Column Count (ID + 120 Breeds)
    expected_cols = 1 + 120
    assert (
        len(sub_df.columns) == expected_cols
    ), f"Expected {expected_cols} columns, got {len(sub_df.columns)}"

    # Verify Probabilities Sum to 1
    # Sum all columns except 'id'
    row_sums = sub_df.iloc[:, 1:].sum(axis=1)
    # Check if all sums are close to 1.0
    valid_probs = np.allclose(row_sums, 1.0, atol=1e-5)
    assert valid_probs, "Probabilities do not sum to 1.0 per row."

    print("    Submission file structure and probability integrity verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
