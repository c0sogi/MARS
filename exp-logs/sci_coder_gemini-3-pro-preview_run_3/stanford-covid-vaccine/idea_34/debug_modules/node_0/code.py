import os
import shutil
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import config
from library.utils import set_seed, format_submission, calculate_mcrmse
from library.data import get_dataloaders
from library.model import DCASGBiGRU
from library.train import MCRMSELoss, train_one_epoch, validate


def main():
    print("Starting RNA Degradation Prediction Demo...")

    # 1. Setup & Configuration Overrides
    # We override config settings to make this a fast demonstration
    print("\n[1] Configuring environment...")
    config.DEBUG = True
    config.EPOCHS = 1
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    config.WORKING_DIR = "./working/demo_run"
    config.BEST_MODEL_PATH = os.path.join(config.WORKING_DIR, "demo_model.pth")
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    # Update cache paths to be inside the demo directory to avoid conflicts
    config.TRAIN_CACHE = os.path.join(config.WORKING_DIR, "train_data.npz")
    config.VAL_CACHE = os.path.join(config.WORKING_DIR, "val_data.npz")
    config.TEST_CACHE = os.path.join(config.WORKING_DIR, "test_data.npz")

    # Ensure demo directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set reproducible seed
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {config.WORKING_DIR}")

    # 2. Data Loading
    print("\n[2] Loading Data (Debug Mode)...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    inputs = batch["inputs"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_masks = batch["pair_masks"].to(device)
    targets = batch["targets"].to(device)
    ids = batch["id"]

    print(f"    Batch Size: {inputs.size(0)}")
    print(f"    Input Shape: {inputs.shape} (Expected: B, 107, 14)")
    print(f"    Target Shape: {targets.shape} (Expected: B, 107, 5)")

    # Assertions for Data
    assert inputs.shape == (config.BATCH_SIZE, 107, 14), "Input shape mismatch"
    assert targets.shape == (config.BATCH_SIZE, 107, 5), "Target shape mismatch"
    assert pair_indices.shape == (config.BATCH_SIZE, 107), "Pair indices shape mismatch"
    assert pair_masks.shape == (config.BATCH_SIZE, 107), "Pair masks shape mismatch"

    # 3. Model Initialization & Forward Pass
    print("\n[3] Initializing Model & Forward Pass...")
    model = DCASGBiGRU().to(device)

    # Run forward pass
    preds = model(inputs, pair_indices, pair_masks)

    print(f"    Prediction Shape: {preds.shape}")

    # Assertions for Model
    assert preds.shape == (
        config.BATCH_SIZE,
        107,
        5,
    ), "Prediction output shape mismatch"
    assert not torch.isnan(preds).any(), "Model produced NaN predictions"

    # 4. Loss Calculation
    print("\n[4] Calculating Loss...")
    criterion = MCRMSELoss()

    # We slice predictions and targets to the scored length (68) as per competition metric logic
    # The loss function inside train.py handles slicing, but here we invoke the class directly
    # to demonstrate the logic. The logic in train_one_epoch does the slicing before passing to criterion.
    preds_sliced = preds[:, : config.SEQ_SCORED, :]
    targets_sliced = targets[:, : config.SEQ_SCORED, :]

    loss = criterion(preds_sliced, targets_sliced)
    print(f"    Initial Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Step...")
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Run one epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, config.MAX_GRAD_NORM
    )
    print(f"    Epoch Training Loss: {train_loss:.6f}")

    assert train_loss > 0, "Training loss should be positive"

    # 6. Validation Demonstration
    print("\n[6] Running Validation...")
    val_score = validate(model, val_loader, device)
    print(f"    Validation MCRMSE: {val_score:.6f}")

    assert val_score > 0, "Validation score should be positive"

    # 7. Submission Generation Demonstration
    print("\n[7] Generating Submission Format...")

    # Simulate predictions for the test set
    # In a real run, we would iterate through test_loader.
    # Here we just take the first batch from test_loader for demonstration.
    test_batch = next(iter(test_loader))
    test_inputs = test_batch["inputs"].to(device)
    test_pair_indices = test_batch["pair_indices"].to(device)
    test_pair_masks = test_batch["pair_masks"].to(device)
    test_ids = test_batch["id"]

    model.eval()
    with torch.no_grad():
        test_preds = model(test_inputs, test_pair_indices, test_pair_masks)
        test_preds_np = test_preds.cpu().numpy()

    # Format submission
    sub_df = format_submission(test_ids, test_preds_np)

    print(f"    Submission DataFrame Shape: {sub_df.shape}")
    print(f"    Columns: {list(sub_df.columns)}")
    print(sub_df.head(2))

    # Assertions for Submission
    expected_rows = len(test_ids) * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"
    assert "id_seqpos" in sub_df.columns, "Missing id_seqpos column"
    assert "reactivity" in sub_df.columns, "Missing target columns"

    # Save submission to verify file writing
    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    print("\n[8] Cleanup...")
    # Clean up the demo working directory
    # shutil.rmtree(config.WORKING_DIR) # Commented out to allow inspection if needed
    print(f"    Demo artifacts stored in {config.WORKING_DIR}")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
