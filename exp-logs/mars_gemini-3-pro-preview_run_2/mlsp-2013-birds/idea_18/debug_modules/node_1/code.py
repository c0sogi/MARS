import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, get_test_loader, make_folds
from library.models import BirdModel
from library.engine import (
    get_pos_weight,
    train_one_epoch,
    validate,
    inference_with_tta,
    save_submission,
)


def run_demo():
    print("=== Starting Bird Species Classification Demo ===")

    # 1. Setup & Configuration
    # We override Config defaults to ensure the demo runs quickly (under 1 hour)
    print("[1/7] Configuring environment...")
    seed_everything(42)

    # Set demo-specific paths and parameters
    Config.WORKING_DIR = "./working/demo_execution"
    Config.BATCH_SIZE = 8  # Small batch size for demonstration
    Config.NUM_WORKERS = 2  # Moderate workers
    Config.DEBUG = True  # Enable debug mode to use data subset
    Config.DEBUG_DATA_SUBSET = 0.1  # Use only 10% of data for speed
    Config.MAX_STEPS = (
        10  # Limit steps (conceptually, though train_one_epoch runs full loader)
    )

    # Ensure directories exist
    Config.setup()

    device = Config.DEVICE
    print(f"      Device: {device}")
    print(f"      Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("[2/7] Initializing Data Loaders...")
    # Generate folds (or load cached) and get loaders for Fold 0
    # This implicitly tests make_folds and BirdDataset
    train_loader, val_loader = get_dataloaders(
        fold_idx=0, debug=Config.DEBUG, batch_size=Config.BATCH_SIZE
    )
    test_loader = get_test_loader(batch_size=Config.BATCH_SIZE)

    print(f"      Train Batches: {len(train_loader)}")
    print(f"      Val Batches:   {len(val_loader)}")
    print(f"      Test Batches:  {len(test_loader)}")

    # Verify Train Batch Structure
    try:
        batch = next(iter(train_loader))
        imgs, labels, rec_ids = batch["image"], batch["labels"], batch["rec_id"]

        print(f"      Sample Batch Shape: Images={imgs.shape}, Labels={labels.shape}")

        # Assertions to verify data pipeline logic
        assert imgs.ndim == 4, "Images should be 4D (B, C, H, W)"
        assert imgs.shape[1] == 3, "Images should have 3 channels (Pseudo-RGB)"
        assert (
            labels.shape[1] == Config.NUM_CLASSES
        ), f"Labels should have {Config.NUM_CLASSES} classes"
        assert labels.dtype == torch.float32, "Labels should be float32"
    except StopIteration:
        raise ValueError(
            "Train loader is empty! Check data paths or debug subset size."
        )

    # 3. Model Initialization
    print("[3/7] Initializing Model...")
    # Use ResNet18 for speed. pretrained=False ensures no network calls/hangs during demo.
    model = BirdModel("resnet18", pretrained=False)
    model.to(device)

    # Verify Forward Pass logic
    with torch.no_grad():
        dummy_input = imgs.to(device)
        dummy_output = model(dummy_input)

    print(f"      Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        imgs.shape[0],
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # 4. Training Loop Execution
    print("[4/7] Running Training Loop (1 Epoch)...")

    # Calculate positive weights for loss function
    # train_loader.dataset.df gives access to the underlying dataframe
    pos_weight = get_pos_weight(train_loader.dataset.df)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # Train for one epoch
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
        pos_weight=pos_weight,
        scheduler=scheduler,
    )

    print(f"      Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss resulted in NaN"

    # 5. Validation Execution
    print("[5/7] Running Validation...")
    val_loss, val_auc = validate(model, val_loader, device, pos_weight)

    print(f"      Val Loss: {val_loss:.4f}")
    print(f"      Val AUC:  {val_auc:.4f}")
    # Note: AUC might be 0.0 if the small debug subset lacks positive samples for some classes,
    # but the function should handle it gracefully without crashing.

    # 6. Inference (Test Time Augmentation)
    print("[6/7] Running Inference with TTA...")
    rec_ids_pred, probs_pred = inference_with_tta(model, test_loader, device)

    print(f"      Predictions Shape: {probs_pred.shape}")
    assert len(rec_ids_pred) == len(
        probs_pred
    ), "Mismatch between IDs and Predictions count"
    assert (
        probs_pred.shape[1] == Config.NUM_CLASSES
    ), "Mismatch in prediction class count"

    # 7. Submission Generation
    print("[7/7] Generating Submission File...")
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    save_submission(rec_ids_pred, probs_pred, sub_path)

    # Verify Submission File
    if os.path.exists(sub_path):
        df_sub = pd.read_csv(sub_path)
        print(f"      Submission Saved to: {sub_path}")
        print(f"      Total Rows: {len(df_sub)}")

        # Check format: Id, Probability
        assert (
            "Id" in df_sub.columns and "Probability" in df_sub.columns
        ), "Submission columns missing"

        # Check row count: num_test_samples * num_classes
        expected_rows = len(rec_ids_pred) * Config.NUM_CLASSES
        assert (
            len(df_sub) == expected_rows
        ), f"Expected {expected_rows} rows, found {len(df_sub)}"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
