import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_folds, get_dataloaders, get_test_loader
from library.models import get_model
from library.engine import train_loop, predict


def run_demo():
    print("==== Starting Demonstration Script ====")

    # 1. Setup Configuration for Demo
    # We override Config attributes to ensure the script runs quickly and strictly
    print("Configuring environment...")

    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 30  # Small subset for speed
    Config.EPOCHS = 2  # Minimal epochs
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup demo-specific paths
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize directories
    Config.setup()

    # Set seeds
    seed_everything(Config.SEED)
    print("Configuration complete.")

    # 2. Verify Data Pipeline
    print("\n==== Verifying Data Pipeline ====")

    # Test Fold Generation
    print("Generating/Loading folds...")
    folds_df = get_folds(load_cached_data=False)  # Force regeneration for demo
    assert "fold" in folds_df.columns, "Folds DataFrame missing 'fold' column"
    print(f"Folds generated. Shape: {folds_df.shape}")

    # Test DataLoaders
    print("Creating DataLoaders (Fold 0, Source: standard)...")
    train_loader, val_loader = get_dataloaders(
        fold_idx=0,
        data_source="standard",
        batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
    )

    # Verify Batch Shapes
    print("Fetching one batch...")
    images, labels = next(iter(train_loader))

    # Expected: (Batch, 3, 224, 224)
    expected_img_shape = (Config.BATCH_SIZE, 3, 224, 224)
    # Expected: (Batch, 19)
    expected_lbl_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    # Handle potential drop_last=True or smaller last batch if debug samples < batch size
    # But with 30 samples and batch 8, first batch should be full.
    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Got {images.shape}, expected {expected_img_shape}"
    assert (
        labels.shape == expected_lbl_shape
    ), f"Label shape mismatch. Got {labels.shape}, expected {expected_lbl_shape}"

    print(f"Batch verification passed. Images: {images.shape}, Labels: {labels.shape}")

    # 3. Verify Model Initialization
    print("\n==== Verifying Model Initialization ====")
    device = Config.DEVICE
    model_name = "resnet18"

    print(f"Instantiating {model_name}...")
    model = get_model(
        model_name, pretrained=True, num_classes=Config.NUM_CLASSES, device=device
    )

    # Check if model is on correct device
    param = next(model.parameters())
    is_cuda = param.is_cuda
    if device == "cuda":
        assert is_cuda, "Model requested on CUDA but parameters are on CPU"

    print("Model instantiated successfully.")

    # 4. Run Training Loop
    print("\n==== Running Training Loop (Demo) ====")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    save_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_fold_0_best.pth")

    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.EPOCHS,
        save_path=save_path,
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved!"
    print(f"Training complete. Checkpoint saved at {save_path}")

    # 5. Inference and Submission
    print("\n==== Running Inference and Submission ====")

    # Load best model for inference
    # We load into a fresh model instance to ensure state dict loading works
    inference_model = get_model(
        model_name, pretrained=False, num_classes=Config.NUM_CLASSES, device=device
    )
    inference_model.load_state_dict(torch.load(save_path, map_location=device))

    # Get Test Loader
    # Note: get_test_loader reads from metadata/test.csv.
    # Since we can't easily debug-subset the test loader via the provided function arguments (it doesn't accept debug arg),
    # we will run it on the full test set (64 samples), which is small enough.
    test_loader = get_test_loader(data_source="standard", batch_size=Config.BATCH_SIZE)

    print("Predicting on test set...")
    preds, rec_ids = predict(inference_model, test_loader, device)

    print(f"Predictions shape: {preds.shape}")
    assert (
        preds.shape[1] == Config.NUM_CLASSES
    ), "Prediction output has incorrect number of classes"
    assert len(preds) == len(rec_ids), "Mismatch between predictions and recording IDs"

    # Format Submission
    # The requirement: Id = rec_id * 100 + species_id
    print("Formatting submission...")

    submission_rows = []
    for i, rid in enumerate(rec_ids):
        probs = preds[i]
        for species_idx, prob in enumerate(probs):
            row_id = int(rid * 100 + species_idx)
            submission_rows.append({"Id": row_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Sort by Id
    submission_df = submission_df.sort_values("Id").reset_index(drop=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("First 5 rows:")
    print(submission_df.head())

    # Final Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"
    loaded_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(loaded_sub.columns) == [
        "Id",
        "Probability",
    ], "Incorrect columns in submission"
    assert len(loaded_sub) > 0, "Submission file is empty"

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
