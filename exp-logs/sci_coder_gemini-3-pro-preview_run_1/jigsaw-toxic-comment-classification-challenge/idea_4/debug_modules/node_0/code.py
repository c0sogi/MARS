import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, logging as transformers_logging

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import load_dataset, make_loader
from library.model import ToxicityModel
from library.engine import train_one_epoch, valid_one_epoch


def run_demo():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print(">>> [1/7] Setting up configuration for demonstration...")

    # Suppress transformers warnings
    transformers_logging.set_verbosity_error()

    # Override Config for speed and isolation
    Config.seed = 999
    Config.debug = True  # Forces data loading to sample small subsets if logic exists
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.working_dir = "./working/demo_execution"

    # Update derived paths in Config (since they are defined at class level)
    os.makedirs(Config.working_dir, exist_ok=True)
    Config.train_cache_path = os.path.join(Config.working_dir, "train_cache.parquet")
    Config.val_cache_path = os.path.join(Config.working_dir, "val_cache.parquet")
    Config.test_cache_path = os.path.join(Config.working_dir, "test_cache.parquet")
    Config.model_save_path = os.path.join(Config.working_dir, "model_demo.pth")

    seed_everything(Config.seed)
    print("    Configuration updated. Working directory:", Config.working_dir)

    # =========================================================================
    # 2. Data Loading & Verification
    # =========================================================================
    print("\n>>> [2/7] Loading and verifying datasets...")

    # Load datasets (load_dataset handles metadata merging)
    # We force reload to ensure we aren't using stale caches for this demo
    if os.path.exists(Config.train_cache_path):
        os.remove(Config.train_cache_path)
    if os.path.exists(Config.val_cache_path):
        os.remove(Config.val_cache_path)
    if os.path.exists(Config.test_cache_path):
        os.remove(Config.test_cache_path)

    train_df = load_dataset("train", load_cached_data=False)
    val_df = load_dataset("val", load_cached_data=False)
    test_df = load_dataset("test", load_cached_data=False)

    # Manually slice to extremely small subsets for the demo to run in seconds
    train_df = train_df.head(20).reset_index(drop=True)
    val_df = val_df.head(10).reset_index(drop=True)
    test_df = test_df.head(10).reset_index(drop=True)

    print(f"    Train shape: {train_df.shape}")
    print(f"    Val shape:   {val_df.shape}")
    print(f"    Test shape:  {test_df.shape}")

    # Assertions
    assert "comment_text" in train_df.columns, "Train DF missing 'comment_text'"
    assert all(
        col in train_df.columns for col in Config.target_cols
    ), "Train DF missing target columns"
    assert "id" in test_df.columns, "Test DF missing 'id'"

    print("    Data integrity checks passed.")

    # =========================================================================
    # 3. DataLoader Creation
    # =========================================================================
    print("\n>>> [3/7] Creating DataLoaders...")

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    train_loader = make_loader(
        train_df, tokenizer, is_train=True, batch_size=Config.train_batch_size
    )
    val_loader = make_loader(
        val_df, tokenizer, is_train=False, batch_size=Config.valid_batch_size
    )
    test_loader = make_loader(
        test_df, tokenizer, is_train=False, batch_size=Config.valid_batch_size
    )

    # Verify batch structure
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    print(f"    Batch keys: {list(batch.keys())}")
    print(f"    Input IDs shape: {input_ids.shape}")
    print(f"    Labels shape: {labels.shape}")

    assert input_ids.shape == (
        Config.train_batch_size,
        Config.max_len,
    ), "Incorrect input_ids shape"
    assert labels.shape == (
        Config.train_batch_size,
        Config.num_classes,
    ), "Incorrect labels shape"
    print("    DataLoader verification passed.")

    # =========================================================================
    # 4. Model Instantiation & Forward Pass
    # =========================================================================
    print("\n>>> [4/7] Initializing Model and checking forward pass...")

    device = Config.device
    model = ToxicityModel()
    model.to(device)

    # Run forward pass with the batch fetched earlier
    with torch.no_grad():
        logits = model(input_ids.to(device), attention_mask.to(device))

    print(f"    Logits shape: {logits.shape}")
    assert logits.shape == (
        Config.train_batch_size,
        Config.num_classes,
    ), "Model output shape mismatch"
    print("    Model forward pass verified.")

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n>>> [5/7] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = None  # Skip scheduler for this short demo

    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
    )

    print(f"    Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss > 0, "Training loss should be positive"

    # =========================================================================
    # 6. Validation & Metric Demonstration
    # =========================================================================
    print("\n>>> [6/7] Running Validation Loop...")

    val_loss, val_preds, val_labels = valid_one_epoch(
        model=model, loader=val_loader, criterion=criterion, device=device
    )

    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Predictions shape: {val_preds.shape}")

    # Calculate Score
    score = get_score(val_labels, val_preds)
    print(f"    ROC AUC Score: {score:.4f}")

    assert val_preds.shape == (
        len(val_df),
        Config.num_classes,
    ), "Validation preds shape mismatch"
    assert 0 <= score <= 1, "Score out of range [0, 1]"

    # =========================================================================
    # 7. Inference Demonstration
    # =========================================================================
    print("\n>>> [7/7] Running Inference on Test Set...")

    # Re-use valid_one_epoch logic or manual inference
    # Since test loader has no labels, we iterate manually to show how to handle it
    model.eval()
    test_preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            probs = torch.sigmoid(logits)
            test_preds_list.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds_list, axis=0)

    print(f"    Test Predictions shape: {test_preds.shape}")
    assert test_preds.shape == (
        len(test_df),
        Config.num_classes,
    ), "Test predictions shape mismatch"

    # Construct submission dataframe
    sub_df = pd.DataFrame(test_preds, columns=Config.target_cols)
    sub_df["id"] = test_df["id"]
    sub_df = sub_df[["id"] + Config.target_cols]

    print("    Sample Submission Head:")
    print(sub_df.head(2))

    print("\n>>> Demonstration Complete. All checks passed successfully.")


if __name__ == "__main__":
    run_demo()
