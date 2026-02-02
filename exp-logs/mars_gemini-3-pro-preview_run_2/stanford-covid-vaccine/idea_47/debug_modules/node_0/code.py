import os
import pandas as pd
import torch
import numpy as np
import shutil

# Import components from the provided library
from library.config import Config
from library.utils import set_seed, format_submission
from library.data import get_dataloaders, RNADataProcessor
from library.model import PFDRN
from library.train import train_epoch, validate, predict


def main():
    print("==== Starting Demo Execution ====")

    # 1. Configuration Overrides for Demo
    # We modify the Config class attributes directly to adapt to the demo environment
    print("[Setup] Configuring environment...")
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = Config.WORKING_DIR
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Reduce hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Set random seed for reproducibility
    set_seed(42)

    # 2. Create Mini Datasets
    # We take a small slice of the provided metadata to create a fast-running subset
    print("[Data] Creating mini datasets from metadata...")

    # Load original metadata
    train_full = pd.read_csv("./metadata/train.csv")
    val_full = pd.read_csv("./metadata/val.csv")
    test_full = pd.read_csv("./metadata/test.csv")

    # Slice top 16 samples for train, 8 for val/test
    mini_train = train_full.head(16)
    mini_val = val_full.head(8)
    mini_test = test_full.head(8)

    # Save mini datasets to working directory
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Update Config to point to these new mini files
    Config.TRAIN_METADATA = mini_train_path
    Config.VAL_METADATA = mini_val_path
    Config.TEST_METADATA = mini_test_path

    # 3. Initialize DataLoaders
    print("[Data] Initializing DataLoaders...")
    # load_cached_data=False ensures we process the new mini CSVs instead of loading old cache
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    # Verify Data Shapes
    sample_batch = next(iter(train_loader))
    inputs = sample_batch["inputs"]
    targets = sample_batch["targets"]

    print(
        f"  Batch Input Shape: {inputs.shape} (Expected: {Config.BATCH_SIZE}, 18, 107)"
    )
    print(
        f"  Batch Target Shape: {targets.shape} (Expected: {Config.BATCH_SIZE}, 107, 5)"
    )

    assert inputs.shape == (Config.BATCH_SIZE, 18, 107), "Input tensor shape mismatch"
    assert targets.shape == (Config.BATCH_SIZE, 107, 5), "Target tensor shape mismatch"

    # 4. Initialize Model
    print("[Model] Initializing PF-DRN Model...")
    device = Config.DEVICE
    model = PFDRN().to(device)

    # Verify Forward Pass
    print("  Verifying forward pass...")
    model.train()
    inputs = inputs.to(device)
    p_idx = sample_batch["partner_indices"].to(device)
    mask = sample_batch["pairing_mask"].to(device)

    # In training mode, model returns (preds_pass1, preds_pass2)
    preds_1, preds_2 = model(inputs, p_idx, mask)

    print(f"  Output Shapes: Pass1 {preds_1.shape}, Pass2 {preds_2.shape}")
    assert preds_1.shape == (Config.BATCH_SIZE, 107, 5)
    assert preds_2.shape == (Config.BATCH_SIZE, 107, 5)

    # 5. Training Loop Demo
    print("[Training] Running one training epoch...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"  Train Loss: {train_loss:.6f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # 6. Validation Demo
    print("[Validation] Running validation...")
    val_score = validate(model, val_loader, device)
    print(f"  Validation MCRMSE: {val_score:.6f}")
    assert val_score >= 0, "Validation score must be non-negative"

    # 7. Inference Demo
    print("[Inference] Generating predictions on test set...")
    test_preds, test_ids = predict(model, test_loader, device)

    print(f"  Prediction Matrix Shape: {test_preds.shape}")
    expected_rows = len(mini_test)
    assert test_preds.shape == (expected_rows, 107, 5), "Prediction shape mismatch"
    assert len(test_ids) == expected_rows, "ID count mismatch"

    # 8. Submission Formatting
    print("[Submission] Formatting submission file...")
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    format_submission(test_ids, test_preds, save_path=submission_path)

    assert os.path.exists(submission_path), "Submission file not created"

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    print(f"  Submission DataFrame Shape: {sub_df.shape}")

    # Expected rows: Number of test samples * Sequence Length (107)
    expected_sub_rows = expected_rows * 107
    assert (
        len(sub_df) == expected_sub_rows
    ), f"Submission row count mismatch. Got {len(sub_df)}, expected {expected_sub_rows}"

    print("==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
