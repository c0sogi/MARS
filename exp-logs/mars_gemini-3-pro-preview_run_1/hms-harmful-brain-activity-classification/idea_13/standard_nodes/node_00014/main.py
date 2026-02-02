import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial

# 1. Suppress tqdm progress bars before importing library modules
import tqdm

tqdm.tqdm = partial(tqdm.tqdm, disable=True)

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, kl_divergence
from library.data import MultiModalDataset
from library.models import AuxiliaryFusionNet
from library.train import train_one_epoch, validate, inference


def run_pipeline():
    # --- Configuration ---
    # Override Config for Fast Baseline
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 4  # Adjust based on CPU availability

    # Ensure output directories exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # --- Data Loading ---
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Initialize Datasets
    # Train on full dataset for 1 epoch to maximize performance within time limit
    train_ds = MultiModalDataset(train_df, mode="train", augment=True)
    val_ds = MultiModalDataset(val_df, mode="val", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model Initialization ---
    model = AuxiliaryFusionNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        anneal_strategy="cos",
    )

    # --- Training ---
    # Train for 1 epoch
    epoch = 0
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, device, epoch
    )

    # Save model
    torch.save(model.state_dict(), Config.MODEL_PATH)

    # --- Validation ---
    # Validate on the entire hold-out set
    val_loss, val_metric = validate(model, val_loader, device)

    print(f"Final Validation Metric: {val_metric}")

    # --- Failure Analysis ---
    # We need predictions and targets to compute per-sample KL
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for eeg, spec, targets in val_loader:
            eeg = eeg.to(device)
            spec = spec.to(device)

            # Forward pass (Joint Head)
            joint_logits, _, _ = model(eeg, spec)
            probs = F.softmax(joint_logits, dim=1)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(targets.numpy())

    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)

    # Calculate KL per sample
    # KL = sum(p * log(p/q))
    epsilon = 1e-15
    y_pred = np.clip(all_preds, epsilon, 1 - epsilon)
    y_true = all_targets

    # Safe log calculation
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = y_true * np.log(y_true / y_pred)
    terms = np.nan_to_num(terms, nan=0.0)
    sample_kl = np.sum(terms, axis=1)

    # Create Analysis DataFrame
    analysis_df = val_df.copy()
    # Ensure length matches (dataloaders might drop last if configured, but val usually doesn't)
    if len(analysis_df) != len(sample_kl):
        analysis_df = analysis_df.iloc[: len(sample_kl)]

    analysis_df["error_kl"] = sample_kl

    # Correlation Analysis
    corr_cols = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]
    print("\nFailure Analysis (Correlation with Error):")
    for col in corr_cols:
        if col in analysis_df.columns:
            corr = analysis_df[col].corr(analysis_df["error_kl"])
            print(f"{col}: {corr}")

    # --- Submission ---
    THRESHOLD = 0.6822116374969482

    if val_metric < THRESHOLD:
        test_df = pd.read_csv(Config.TEST_CSV)
        test_ds = MultiModalDataset(test_df, mode="test", augment=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        test_preds = inference(model, test_loader, device)

        # Align lengths
        if len(test_df) != len(test_preds):
            test_df = test_df.iloc[: len(test_preds)]

        submission_df = pd.DataFrame(
            {
                "eeg_id": test_df["eeg_id"],
                "seizure_vote": test_preds[:, 0],
                "lpd_vote": test_preds[:, 1],
                "gpd_vote": test_preds[:, 2],
                "lrda_vote": test_preds[:, 3],
                "grda_vote": test_preds[:, 4],
                "other_vote": test_preds[:, 5],
            }
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_pipeline()
