import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import logging as transformers_logging

# Import from the provided library files
from library.config import Config, seed_everything
from library.utils import compute_qwk, optimize_thresholds, apply_thresholds
from library.data_loader import get_dataloaders
from library.modeling import EssayRegressor
from library.engine import train_fn, eval_fn

# Suppress verbose transformer warnings
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    print("=== Starting Essay Scoring Library Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup for Rapid Demo
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for rapid execution...")
    seed_everything(42)

    # Create a temporary directory for this demo run
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config attributes to run a lightweight version of the task
    Config.WORKING_DIR = DEMO_DIR
    Config.METADATA_DIR = "./metadata"
    Config.MAX_LENGTH = 64  # Short context for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Main process only to avoid overhead
    Config.GRAD_ACCUMULATION_STEPS = 1  # Simplify for demo

    # -------------------------------------------------------------------------
    # 2. Data Preparation (Tiny Subset)
    # -------------------------------------------------------------------------
    print("[2] Creating tiny dataset subsets from metadata...")

    # Load original metadata
    full_train = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    full_val = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    full_test = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Create subsets: 20 train, 10 val, 10 test
    demo_train = full_train.head(20).copy()
    demo_val = full_val.head(10).copy()
    demo_test = full_test.head(10).copy()

    # Save to demo directory
    Config.TRAIN_PATH = os.path.join(DEMO_DIR, "train.csv")
    Config.VAL_PATH = os.path.join(DEMO_DIR, "val.csv")
    Config.TEST_PATH = os.path.join(DEMO_DIR, "test.csv")

    demo_train.to_csv(Config.TRAIN_PATH, index=False)
    demo_val.to_csv(Config.VAL_PATH, index=False)
    demo_test.to_csv(Config.TEST_PATH, index=False)

    print(f"    Train subset shape: {demo_train.shape}")
    print(f"    Val subset shape:   {demo_val.shape}")
    print(f"    Test subset shape:  {demo_test.shape}")

    # -------------------------------------------------------------------------
    # 3. Utility Function Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Utility Functions (QWK & Thresholds)...")

    # Test QWK
    y_true = np.array([1, 2, 3, 4, 5, 6])
    y_pred_perfect = np.array([1, 2, 3, 4, 5, 6])
    score = compute_qwk(y_true, y_pred_perfect)
    assert np.isclose(score, 1.0), f"QWK for perfect match should be 1.0, got {score}"
    print("    compute_qwk: Passed")

    # Test Threshold Optimization
    # Create continuous predictions slightly offset from integers
    y_pred_cont = np.array([1.1, 1.9, 3.1, 4.2, 4.8, 5.9])
    opt_thresholds = optimize_thresholds(y_true, y_pred_cont)

    print(f"    Optimized Thresholds: {opt_thresholds}")
    assert len(opt_thresholds) == 5, "Should return exactly 5 thresholds"
    assert np.all(np.diff(opt_thresholds) >= 0), "Thresholds must be sorted"

    # Test Threshold Application
    y_pred_disc = apply_thresholds(y_pred_cont, opt_thresholds)
    assert np.array_equal(
        y_pred_disc, y_true
    ), "Threshold application failed to recover labels"
    print("    optimize_thresholds & apply_thresholds: Passed")

    # -------------------------------------------------------------------------
    # 4. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Data Loaders...")

    # Initialize dataloaders (force processing by ignoring cache)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    print(f"    Batch Input Shape: {input_ids.shape}")
    print(f"    Batch Labels Shape: {labels.shape}")

    assert input_ids.shape == (
        Config.BATCH_SIZE,
        Config.MAX_LENGTH,
    ), "Incorrect input shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label shape"
    assert labels.dtype == torch.float, "Labels should be float for regression"
    print("    Data Loading: Passed")

    # -------------------------------------------------------------------------
    # 5. Model & Engine Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model and Training Engine...")

    device = Config.DEVICE
    print(f"    Device: {device}")

    # Initialize Model
    model = EssayRegressor(Config.MODEL_NAME, pretrained=True)
    model.to(device)

    # Optimizer & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    # 5a. Train Function (One Epoch)
    print("    Running training epoch...")
    train_loss = train_fn(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        device=device,
        scheduler=None,  # No scheduler for demo
        criterion=criterion,
    )
    print(f"    Training Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # 5b. Eval Function (Validation)
    print("    Running validation inference...")
    val_preds, val_loss = eval_fn(
        model=model, data_loader=val_loader, device=device, criterion=criterion
    )
    print(f"    Validation Loss: {val_loss:.4f}")
    assert len(val_preds) == len(demo_val), "Validation predictions count mismatch"
    assert val_preds.shape == (len(demo_val), 1), "Validation output shape mismatch"

    # -------------------------------------------------------------------------
    # 6. Post-Processing & Submission Simulation
    # -------------------------------------------------------------------------
    print("\n[6] Simulating Post-Processing and Submission...")

    # Flatten predictions
    val_preds_flat = val_preds.flatten()
    val_labels_flat = demo_val["score"].values

    # Optimize thresholds based on validation results
    best_thresholds = optimize_thresholds(val_labels_flat, val_preds_flat)

    # Predict on Test Set
    test_preds, _ = eval_fn(
        model=model, data_loader=test_loader, device=device, criterion=None
    )
    test_preds_flat = test_preds.flatten()

    # Apply thresholds to get final scores (1-6)
    final_scores = apply_thresholds(test_preds_flat, best_thresholds)

    # Generate Submission DataFrame
    submission = pd.DataFrame(
        {"essay_id": demo_test["essay_id"], "score": final_scores}
    )

    print("    Sample Submission:")
    print(submission.to_string(index=False))

    # Final Assertions
    assert submission.shape == (10, 2), "Submission shape mismatch"
    assert submission["score"].min() >= 1, "Score below 1 found"
    assert submission["score"].max() <= 6, "Score above 6 found"
    assert not submission.isnull().values.any(), "NaN values in submission"

    print("\n=== Demonstration Completed Successfully ===")
