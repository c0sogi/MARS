import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders
from library.models import get_model
from library.engine import run_two_phase_training
from library.inference import predict_with_tta, save_submission

# -------------------------------------------------------------------------
# Main Execution Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class attributes directly to run a fast demo.
    print(">>> Setting up configuration for fast demonstration...")

    # Reduce epochs to 1 per phase
    Config.EPOCHS_HEAD = 1
    Config.EPOCHS_FINE = 1

    # Reduce batch size for the demo
    Config.BATCH_SIZE = 4

    # Use a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    logger = get_logger("demo_script")
    logger.info("Configuration updated for demo run.")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n>>> Initializing DataLoaders with a subset of data...")

    # We use debug_subset_size=16 to load only 16 images for train/val/test
    # This ensures the dataloaders are created almost instantly.
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debug script to avoid multiprocessing overhead
        load_cached_data=False,  # Force reprocessing to verify logic
        debug_subset_size=16,
    )

    # Validation: Check DataLoader integrity
    print("Validating DataLoader shapes...")
    images, labels = next(iter(train_loader))

    # Check Image Shape: (Batch, 3, 224, 224)
    expected_shape = (Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    if images.shape != expected_shape:
        raise AssertionError(
            f"Expected image shape {expected_shape}, got {images.shape}"
        )

    # Check Label Shape: (Batch,)
    if labels.shape != (Config.BATCH_SIZE,):
        raise AssertionError(
            f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"
        )

    logger.info(f"DataLoaders verified. Classes: {len(class_names)}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n>>> Initializing Model...")

    # Pick the first architecture from the config
    model_name = Config.MODEL_ARCHS[0]
    device = Config.DEVICE

    model = get_model(
        model_name, num_classes=len(class_names), device=device, pretrained=True
    )

    # Validation: Dummy Forward Pass
    print("Validating model forward pass...")
    dummy_input = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    # Check Output Shape: (Batch, Num_Classes)
    if output.shape != (Config.BATCH_SIZE, len(class_names)):
        raise AssertionError(
            f"Expected output shape {(Config.BATCH_SIZE, len(class_names))}, got {output.shape}"
        )

    logger.info(f"Model {model_name} initialized and verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n>>> Running Two-Phase Training (Head Adaptation + Fine-Tuning)...")

    model_save_path = os.path.join(Config.WORKING_DIR, "model_demo.pth")

    # This function handles freezing/unfreezing and the training loop
    # Since we set epochs to 1, this will run very quickly on the subset.
    trained_model = run_two_phase_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=model_save_path,
    )

    # Validation: Check if model checkpoint exists
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_save_path}")

    logger.info("Training simulation completed successfully.")

    # -------------------------------------------------------------------------
    # 5. Inference
    # -------------------------------------------------------------------------
    print("\n>>> Performing Inference on Test Set...")

    ids, probs = predict_with_tta(trained_model, test_loader, device=device)

    # Validation: Check predictions shape
    # We loaded 16 test samples
    expected_samples = 16
    if len(ids) != expected_samples:
        raise AssertionError(f"Expected {expected_samples} IDs, got {len(ids)}")

    if probs.shape != (expected_samples, len(class_names)):
        raise AssertionError(
            f"Expected prob shape {(expected_samples, len(class_names))}, got {probs.shape}"
        )

    # Check probability validity (sum to 1 approx)
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise AssertionError("Probabilities do not sum to 1.")

    logger.info("Inference completed.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n>>> Generating Submission File...")

    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    save_submission(ids, probs, class_names, output_path=submission_path)

    # Validation: Check CSV file
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check columns: id + 120 breeds
    expected_cols = 1 + len(class_names)
    if df_sub.shape[1] != expected_cols:
        raise AssertionError(
            f"Expected {expected_cols} columns, found {df_sub.shape[1]}"
        )

    # Check if IDs match
    if not df_sub["id"].equals(pd.Series(ids)):
        raise AssertionError("IDs in submission do not match inference IDs.")

    print("\n>>> Demo Script Completed Successfully!")
    print(f"Outputs are stored in: {Config.WORKING_DIR}")
