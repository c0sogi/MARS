import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_score, AverageMeter
from library.data_processing import (
    load_data,
    get_structural_features,
    MLMDataset,
    InsultDataset,
)
from library.model import HybridDeberta
from library.awp import AWP
from library.trainer import train_fn


def run_demo():
    print("Starting Demo Run...")

    # ====================================================
    # 1. Configuration Override for Speed
    # ====================================================
    print("Configuring environment for demo...")
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Reduce complexity for speed
    Config.MLM_EPOCHS = 1
    Config.CLS_EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.SVD_COMPONENTS = 10  # Reduced to fit small sample size
    Config.MAX_LEN = 32  # Shorten sequence length for speed
    Config.AWP_START_EPOCH = 0  # Force AWP to run immediately for testing

    seed_everything(Config.SEED)

    # ====================================================
    # 2. Data Loading & Subsetting
    # ====================================================
    print("Loading and subsetting data...")
    train_df, val_df, test_df = load_data()

    # Take a small subset (e.g., 20 samples) to ensure speed
    # We need enough samples for SVD (n_samples >= n_components)
    subset_size = 20
    train_subset = train_df.head(subset_size).copy()
    val_subset = val_df.head(subset_size).copy()
    test_subset = test_df.head(subset_size).copy()

    print(
        f"Subset sizes -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
    )

    # ====================================================
    # 3. Feature Engineering Verification
    # ====================================================
    print("Verifying Structural Feature Generation...")
    # Force re-computation by setting load_cached_data=False (or ensuring cache doesn't exist for this run)
    # Since we changed SVD_COMPONENTS, we shouldn't load old cache.
    # We will mock the cache paths in Config temporarily or just rely on the function logic.
    # The function writes to Config.WORKING_DIR, which we changed above.

    train_svd, val_svd, test_svd = get_structural_features(
        train_subset["Comment"].tolist(),
        val_subset["Comment"].tolist(),
        test_subset["Comment"].tolist(),
        load_cached_data=False,
    )

    # Assertions
    assert train_svd.shape == (
        subset_size,
        Config.SVD_COMPONENTS,
    ), f"Train SVD shape mismatch. Expected ({subset_size}, {Config.SVD_COMPONENTS}), got {train_svd.shape}"
    assert val_svd.shape == (
        subset_size,
        Config.SVD_COMPONENTS,
    ), f"Val SVD shape mismatch. Expected ({subset_size}, {Config.SVD_COMPONENTS}), got {val_svd.shape}"

    print("Structural features generated and verified successfully.")

    # ====================================================
    # 4. Dataset Logic Verification
    # ====================================================
    print("Verifying Dataset Classes...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # A. MLM Dataset
    mlm_dataset = MLMDataset(
        train_subset["Comment"].tolist(), tokenizer, max_len=Config.MAX_LEN
    )
    mlm_item = mlm_dataset[0]

    assert "input_ids" in mlm_item
    assert "labels" in mlm_item
    assert mlm_item["input_ids"].shape == (Config.MAX_LEN,)
    # Check that masking logic happened (not strictly guaranteed for every sample if prob is low, but likely)
    # We just check tensor types here.
    assert isinstance(mlm_item["labels"], torch.Tensor)
    print("MLMDataset verified.")

    # B. Insult Dataset
    insult_dataset = InsultDataset(
        train_subset["Comment"].tolist(),
        train_svd,
        tokenizer,
        max_len=Config.MAX_LEN,
        labels=train_subset["Insult"].values,
    )
    insult_item = insult_dataset[0]

    assert "input_ids" in insult_item
    assert "svd_features" in insult_item
    assert "label" in insult_item
    assert insult_item["svd_features"].shape == (Config.SVD_COMPONENTS,)
    assert insult_item["label"].numel() == 1
    print("InsultDataset verified.")

    # ====================================================
    # 5. Model Architecture Verification
    # ====================================================
    print("Verifying HybridDeberta Model...")
    model = HybridDeberta(pretrained_model_name_or_path=Config.MODEL_NAME)
    model.to(Config.DEVICE)

    # Create a dummy batch
    batch_size = 2
    dummy_input_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, Config.MAX_LEN)
    ).to(Config.DEVICE)
    dummy_mask = torch.ones((batch_size, Config.MAX_LEN)).to(Config.DEVICE)
    dummy_svd = torch.randn((batch_size, Config.SVD_COMPONENTS)).to(Config.DEVICE)

    # Forward pass
    outputs = model(dummy_input_ids, dummy_mask, dummy_svd)

    assert outputs.shape == (
        batch_size,
    ), f"Model output shape mismatch. Expected ({batch_size},), got {outputs.shape}"
    print("Model forward pass verified.")

    # ====================================================
    # 6. Training Loop & AWP Simulation
    # ====================================================
    print("Verifying Training Step with AWP...")

    # Prepare DataLoader
    train_loader = DataLoader(
        insult_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True
    )

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=10
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    # Initialize AWP
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.AWP_LR,
        adv_eps=Config.AWP_EPS,
        start_epoch=Config.AWP_START_EPOCH,
        scaler=scaler,
    )

    # Run one epoch (which is just a few batches due to subset)
    # We use train_fn from library.trainer
    epoch_loss = train_fn(
        train_loader,
        model,
        criterion,
        optimizer,
        epoch=0,
        scheduler=scheduler,
        device=Config.DEVICE,
        awp=awp,
        scaler=scaler,
    )

    print(f"Training step completed. Loss: {epoch_loss:.4f}")
    assert not np.isnan(epoch_loss), "Training loss is NaN"

    # ====================================================
    # 7. Metric Verification
    # ====================================================
    print("Verifying Metric Calculation...")
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    score = get_score(y_true, y_pred)

    assert score == 1.0, f"Metric calculation failed. Expected 1.0, got {score}"
    print(f"Metric verified. AUC: {score}")

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    run_demo()
