import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.data import get_dataloaders, get_test_loader
from library.model import CalibratedSequenceNetwork
from library.loss import WeightedMultiLabelLogLoss
from library.train import run_training


def main():
    print("=== RSNA Cervical Spine Fracture Detection: Pipeline Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_execution"
    Config.BATCH_SIZE = 2  # Reduce batch size for safety during demo
    Config.ACCUMULATION_STEPS = 1
    Config.NUM_WORKERS = 2

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Get dataloaders with debug=True (loads only 10 samples)
    train_loader, val_loader = get_dataloaders(debug=True, load_cached_data=False)

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Targets Shape: {targets.shape}")

    # Assertions
    # Expected: (Batch, Seq_Len, Channels, H, W) -> (2, 96, 3, 256, 256)
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, 3, 256, 256)
    assert (
        images.shape == expected_shape
    ), f"Expected images shape {expected_shape}, got {images.shape}"

    # Expected: (Batch, Num_Classes) -> (2, 8)
    expected_targets = (Config.BATCH_SIZE, Config.NUM_CLASSES)
    assert (
        targets.shape == expected_targets
    ), f"Expected targets shape {expected_targets}, got {targets.shape}"

    print("    Data loading logic verified.")

    # -------------------------------------------------------------------------
    # 3. Model & Loss Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Forward Pass and Loss...")

    device = Config.DEVICE
    model = CalibratedSequenceNetwork().to(device)
    criterion = WeightedMultiLabelLogLoss()

    # Move batch to device
    images = images.to(device)
    targets = targets.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(images)

    print(f"    Logits Shape: {logits.shape}")

    # Assertions
    assert (
        logits.shape == expected_targets
    ), f"Expected logits shape {expected_targets}, got {logits.shape}"

    # Loss calculation
    loss = criterion(logits, targets)
    print(f"    Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("    Model and Loss logic verified.")

    # Clean up GPU memory
    del images, targets, logits, loss, model
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (run_training)...")

    # run_training handles the loop, logging, and saving to Config.WORKING_DIR
    # It uses the Config class we modified earlier.
    run_training(debug=True)

    # Verify output files exist
    expected_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    expected_log_path = os.path.join(Config.WORKING_DIR, "train.log")

    if os.path.exists(expected_model_path):
        print(f"    Success: Model checkpoint found at {expected_model_path}")
    else:
        raise FileNotFoundError(f"Training failed to produce {expected_model_path}")

    if os.path.exists(expected_log_path):
        print(f"    Success: Training log found at {expected_log_path}")
    else:
        raise FileNotFoundError(f"Training failed to produce {expected_log_path}")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Inference and Submission Generation...")

    # Load the model we just trained
    model = CalibratedSequenceNetwork().to(device)
    load_checkpoint(model, expected_model_path, device=device)
    model.eval()

    # Get Test Loader (using metadata logic)
    # Note: The test set in the competition is hidden. The provided test.csv is a placeholder.
    # We will use the provided test_metadata.csv.
    test_loader = get_test_loader(load_cached_data=False, debug=Config.DEBUG)

    predictions = []
    row_ids = []

    print(f"    Inference on {len(test_loader.dataset)} studies...")

    with torch.no_grad():
        for images, study_ids in test_loader:
            images = images.to(device)

            # Forward
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Map predictions to submission format
            # Output columns: C1, C2, C3, C4, C5, C6, C7, patient_overall
            # Submission rows: [StudyID]_[TargetName]

            for i, study_id in enumerate(study_ids):
                # Per study, we have 8 predictions
                study_probs = probs[i]

                for col_idx, target_name in enumerate(Config.TARGET_COLUMNS):
                    row_id = f"{study_id}_{target_name}"
                    prob = study_probs[col_idx]

                    row_ids.append(row_id)
                    predictions.append(prob)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"row_id": row_ids, "fractured": predictions})

    print(f"    Generated {len(submission_df)} predictions.")
    print(f"    Sample rows:\n{submission_df.head(3)}")

    # Save submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"    Submission saved to {submission_path}")

    # Verify Submission Structure
    # Should have columns 'row_id' and 'fractured'
    assert "row_id" in submission_df.columns
    assert "fractured" in submission_df.columns
    assert len(submission_df) > 0

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
