import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, mixup_data, calculate_multilabel_auc
from library.data import get_dataloaders, BirdDataset, get_transforms
from library.model import get_bird_model
from library.engine import (
    train_one_epoch,
    evaluate,
    run_inference,
    fit,
    save_submission,
)


def main():
    print("=== Starting Demonstration of Bird Species Classification Pipeline ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Use only 32 samples
    Config.BATCH_SIZE = 8  # Small batch size
    Config.EPOCHS = 2  # Only 2 epochs
    Config.SWA_START_EPOCH = 1  # Start SWA at epoch 1 (active for the 2nd epoch)
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.PRETRAINED = False  # Skip downloading weights for this demo

    # Set output directory for demo
    Config.WORK_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = Config.WORK_DIR

    # Create directories manually since Config.setup() isn't called explicitly here
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set Seed
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Initialize Loaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Fetch a single batch from training loader
    try:
        images, labels, rec_ids = next(iter(train_loader))
        print(f"    Train Batch - Images: {images.shape}, Labels: {labels.shape}")
    except StopIteration:
        raise Exception("Train loader is empty! Check data paths or subset size.")

    # Assertions for Data Shapes
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Image shape mismatch. Expected {(Config.BATCH_SIZE, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Label shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {labels.shape}"
    assert len(rec_ids) == Config.BATCH_SIZE, "Rec ID count mismatch."

    # Fetch a batch from test loader (different signature: images, rec_ids)
    test_images, test_ids = next(iter(test_loader))
    assert test_images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), "Test image shape mismatch."

    print("    Data loading and shape verification successful.")

    # ---------------------------------------------------------
    # 3. Utility Functions Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Utility Functions...")

    # Test Mixup
    mixed_x, y_a, y_b, lam = mixup_data(
        images, labels, alpha=0.2, device=torch.device("cpu")
    )
    assert mixed_x.shape == images.shape, "Mixup output shape mismatch."
    assert y_a.shape == labels.shape, "Mixup label A shape mismatch."
    assert 0 <= lam <= 1, "Mixup lambda out of range."
    print("    Mixup function works correctly.")

    # Test AUC Calculation
    # Create dummy perfect predictions
    dummy_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    dummy_pred = np.array([[0.9, 0.1, 0.8], [0.2, 0.8, 0.1], [0.95, 0.85, 0.05]])
    auc_score = calculate_multilabel_auc(dummy_true, dummy_pred)
    assert 0.0 <= auc_score <= 1.0, "AUC score out of bounds."
    print(f"    Calculated Dummy AUC: {auc_score:.4f}")

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("\n[4] Initializing Model...")
    model = get_bird_model(pretrained=Config.PRETRAINED).to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images.to(device)
        output = model(dummy_input)

    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {output.shape}"
    print("    Model initialized and forward pass verified.")

    # ---------------------------------------------------------
    # 5. Training Engine Verification
    # ---------------------------------------------------------
    print("\n[5] Testing Training Loop (Engine)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Test: Train One Epoch
    print("    Running 'train_one_epoch'...")
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=0)
    print(f"    > Epoch 0 Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN."

    # Test: Evaluate
    print("    Running 'evaluate'...")
    val_auc, val_loss = evaluate(model, val_loader, device)
    print(f"    > Validation AUC: {val_auc:.4f}, Loss: {val_loss:.4f}")

    # Test: Full Fit Function (with SWA)
    print("    Running 'fit' (2 epochs, SWA enabled)...")
    # Re-initialize model to start fresh
    model = get_bird_model(pretrained=Config.PRETRAINED).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    final_model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=Config.EPOCHS,
        swa_start_epoch=Config.SWA_START_EPOCH,
        patience=2,
    )

    assert final_model is not None, "Fit function returned None."
    print("    Training loop completed successfully.")

    # ---------------------------------------------------------
    # 6. Inference and Submission
    # ---------------------------------------------------------
    print("\n[6] Generating Submission...")

    # Run Inference
    predictions = run_inference(final_model, test_loader, device, tta=True)

    # Verify Prediction Keys
    sample_rec_id = list(predictions.keys())[0]
    assert (
        len(predictions[sample_rec_id]) == Config.NUM_CLASSES
    ), "Prediction vector length mismatch."

    # Save Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    save_submission(predictions, submission_path)

    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify CSV Format
    df_sub = pd.read_csv(submission_path)
    print(f"    Submission saved to {submission_path}")
    print(f"    Submission shape: {df_sub.shape}")

    assert list(df_sub.columns) == ["Id", "Probability"], "Submission columns mismatch."
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
