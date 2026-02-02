import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.data import get_data_loaders
from library.model import GranularSiameseDeBERTa
from library.train import train_fn, eval_fn


def run_demonstration():
    print("======================================================")
    print("       Starting Library Usage Demonstration           ")
    print("======================================================")

    # 1. Setup & Configuration Overrides
    # --------------------------------------------------------
    # We override Config parameters to run a fast demo on a small subset.
    print("\n[1] Configuring environment for demo...")

    DEMO_DIR = "./working/demo_task"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Monkey-patch Config
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_PATH = os.path.join(DEMO_DIR, "train.csv")
    Config.VAL_PATH = os.path.join(DEMO_DIR, "val.csv")
    Config.TEST_PATH = os.path.join(DEMO_DIR, "test.csv")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated. Random seed set.")

    # 2. Prepare Data Subsets
    # --------------------------------------------------------
    print("\n[2] Creating data subsets from metadata...")

    # Read original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create tiny subsets (20 train, 8 val, 8 test) to ensure batches work
    subset_train = orig_train.head(20).copy()
    subset_val = orig_val.head(8).copy()
    subset_test = orig_test.head(8).copy()

    # Save to demo directory
    subset_train.to_csv(Config.TRAIN_PATH, index=False)
    subset_val.to_csv(Config.VAL_PATH, index=False)
    subset_test.to_csv(Config.TEST_PATH, index=False)

    print(
        f"    Saved subsets: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
    )

    # 3. Data Loading & Processing
    # --------------------------------------------------------
    print("\n[3] Testing Data Loading and Processing...")

    # Force processing from scratch (load_cached_data=False)
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=False)

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    required_keys = [
        "q_input_ids",
        "q_attention_mask",
        "q_segment_ids",
        "a_input_ids",
        "a_attention_mask",
        "cats",
        "labels",
    ]

    print("    Verifying batch structure...")
    for key in required_keys:
        if key not in batch:
            raise AssertionError(f"Missing key in batch: {key}")
        print(f"      - {key}: {batch[key].shape}")

    # Assertions on shapes
    assert batch["q_input_ids"].shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)
    assert batch["labels"].shape == (Config.TRAIN_BATCH_SIZE, 30)
    assert batch["cats"].shape == (Config.TRAIN_BATCH_SIZE, 2)

    # Verify q_segment_ids logic (should contain 0, 1, 2)
    unique_segments = torch.unique(batch["q_segment_ids"])
    print(f"    Unique segment IDs found: {unique_segments.tolist()}")

    print("    Data loading verification passed.")

    # 4. Model Initialization & Forward Pass
    # --------------------------------------------------------
    print("\n[4] Testing Model Initialization and Forward Pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")

    model = GranularSiameseDeBERTa()
    model.to(device)

    # Move batch to device
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

    # Forward pass
    model.eval()
    with torch.no_grad():
        preds = model(**inputs)

    print(f"    Prediction shape: {preds.shape}")

    # Assertions
    assert preds.shape == (
        Config.TRAIN_BATCH_SIZE,
        30,
    ), f"Expected shape ({Config.TRAIN_BATCH_SIZE}, 30), got {preds.shape}"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions must be in range [0, 1] (Sigmoid output)"

    print("    Model forward pass verification passed.")

    # 5. Training Loop Demonstration
    # --------------------------------------------------------
    print("\n[5] Testing Training Loop (1 Epoch)...")

    # Setup Optimizer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": Config.LR_BACKBONE,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": Config.LR_HEAD,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1, num_training_steps=5
    )

    # Run Train Function
    train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch=0)
    print(f"    Training complete. Average Loss: {train_loss:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss > 0, "Training loss should be positive"

    # 6. Evaluation & Metric
    # --------------------------------------------------------
    print("\n[6] Testing Evaluation and Metric Calculation...")

    val_loss, val_preds, val_targets = eval_fn(model, val_loader, device)

    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Val Preds Shape: {val_preds.shape}")
    print(f"    Val Targets Shape: {val_targets.shape}")

    assert val_preds.shape == val_targets.shape
    assert len(val_preds) == len(subset_val)

    # Compute Metric
    score = compute_spearmanr(val_preds, val_targets)
    print(f"    Spearman Correlation Score: {score:.4f}")

    assert -1.0 <= score <= 1.0, "Spearman score out of range [-1, 1]"

    print("    Evaluation verification passed.")

    print("\n======================================================")
    print("       Demonstration Completed Successfully           ")
    print("======================================================")


if __name__ == "__main__":
    run_demonstration()
