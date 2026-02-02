import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import importlib

# Import provided library modules
from library import config

importlib.reload(config)
from library import utils

importlib.reload(utils)
from library import data

importlib.reload(data)
from library import model

importlib.reload(model)
from library import train

importlib.reload(train)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override config for this run
    config.EPOCHS = 3
    config.DEBUG = False

    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    # Setup Device
    device = torch.device(config.DEVICE)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    train_loader, val_loader, test_loader = data.get_dataloaders(
        train_batch_size=config.BATCH_SIZE,
        val_batch_size=config.BATCH_SIZE,
        debug=config.DEBUG,
        load_cached_data=True,
    )

    # ==========================================
    # 3. Model & Optimizer Initialization
    # ==========================================
    net = model.EEGWaveNet(pretrained=True)
    net.to(device)

    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # Loss function for training (batchmean reduction)
    criterion = utils.KL_Loss(reduction="batchmean")

    # Mixed Precision Scaler
    scaler = GradScaler()

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_loss = float("inf")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train Step
        train_loss = train.train_one_epoch(
            net, train_loader, optimizer, criterion, device, scaler
        )

        # Validation Step
        val_loss = train.validate(net, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        # Save Checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), config.MODEL_PATH)

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    # Load best model
    net.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    net.eval()

    val_losses = []

    # Load Validation Metadata for Analysis
    val_df = pd.read_csv(config.VAL_CSV)
    if config.DEBUG:
        val_df = val_df.head(config.DEBUG_SIZE)

    # Compute per-sample loss
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = net(inputs)
            log_probs = F.log_softmax(logits, dim=1)

            # Compute KL Divergence per sample
            # reduction='none' returns (Batch, Classes)
            kl_batch = F.kl_div(log_probs, targets, reduction="none")
            # Sum over classes to get scalar KL divergence per sample
            kl_sample = kl_batch.sum(dim=1)

            val_losses.extend(kl_sample.cpu().numpy())

    # Calculate and Print Final Metric
    final_metric = np.mean(val_losses)
    print(f"Final Validation Metric: {final_metric}")

    # Correlation Analysis
    val_df["error"] = val_losses

    # Identify columns for correlation
    numeric_cols = [
        "total_votes",
        "eeg_label_offset_seconds",
        "spectogram_label_offset_seconds",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    available_cols = [c for c in numeric_cols if c in val_df.columns]

    correlations = val_df[available_cols + ["error"]].corr()["error"].drop("error")

    print("Error Correlations with Metadata:")
    print(correlations.sort_values(ascending=False))

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 1.0081

    if final_metric < THRESHOLD:
        # Generate predictions using the helper in train.py
        train.predict(test_loader, device, debug=config.DEBUG)
    else:
        print(
            f"Metric {final_metric} is not below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
