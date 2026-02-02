import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
import library.config as config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    calculate_roc_auc,
)
from library.dataset import get_dataloaders, get_processed_metadata
from library.model import HybridEfficientNet
from library.train import train_one_epoch, validate, inference


def main():
    # 1. Setup
    seed_everything(config.SEED)
    print(f"Running on device: {config.DEVICE}")

    # 2. Data Loading
    print("Loading dataloaders...")
    train_loader, val_loader, test_loader, meta_dim = get_dataloaders(
        load_cached_data=True
    )

    # Load processed metadata dataframe for failure analysis (to get feature names/values)
    # We ignore the returned train/test dfs here, just need val
    _, df_val_meta, _, _ = get_processed_metadata(load_cached_data=True)

    # 3. Model Initialization
    print(f"Initializing HybridEfficientNet with meta_dim={meta_dim}...")
    model = HybridEfficientNet(meta_dim=meta_dim)
    model.to(config.DEVICE)

    # 4. Training Components
    # Use dampened positive weight as per config
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # 5. Training Loop
    best_score = 0.0
    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_one_epoch(
            train_loader, model, criterion, optimizer, config.DEVICE
        )

        # Validate
        val_loss, val_auc = validate(val_loader, model, criterion, config.DEVICE)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} [{elapsed:.0f}s]: "
            f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        # Save Best
        if val_auc > best_score:
            best_score = val_auc
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                },
                is_best=True,
            )
            print(f"  New best score: {best_score:.6f}")

    # 6. Load Best Model for Analysis and Inference
    print("\nLoading best model for analysis...")
    load_checkpoint(model, filename="best_model.pth")

    # 7. Failure Analysis & Final Metric Calculation
    print("Performing failure analysis on validation set with TTA...")
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(config.DEVICE)
            meta = batch["meta"].to(config.DEVICE)
            targets = batch["target"].to(config.DEVICE)

            # Apply TTA (Cite solution_lesson_node_00012)
            logits1 = model(images, meta)
            logits2 = model(torch.flip(images, [2]), meta)
            logits3 = model(torch.flip(images, [3]), meta)
            logits4 = model(torch.flip(images, [2, 3]), meta)

            probs = (
                torch.sigmoid(logits1)
                + torch.sigmoid(logits2)
                + torch.sigmoid(logits3)
                + torch.sigmoid(logits4)
            ) / 4.0

            probs = probs.cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(targets_np)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    final_metric = calculate_roc_auc(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Calculate Error Correlations
    errors = np.abs(val_targets - val_preds)

    # Identify metadata columns (numerical and one-hot encoded)
    # The df_val_meta contains 'meta_' columns which correspond to the input of the tabular branch
    meta_cols = [c for c in df_val_meta.columns if c.startswith("meta_")]

    print("\nCorrelation between Error Magnitude and Input Features:")
    correlations = []

    # Ensure alignment: df_val_meta should align with val_loader (shuffle=False)
    if len(df_val_meta) == len(errors):
        for col in meta_cols:
            feat_values = df_val_meta[col].values
            # Calculate correlation if variance is non-zero
            if np.std(feat_values) > 1e-9 and np.std(errors) > 1e-9:
                corr = np.corrcoef(feat_values, errors)[0, 1]
                correlations.append((col, corr))
            else:
                correlations.append((col, 0.0))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, corr in correlations[:5]:
            print(f"  {name}: {corr:.4f}")
    else:
        print(
            f"  Skipping correlation analysis due to length mismatch: DF {len(df_val_meta)} vs Preds {len(errors)}"
        )

    # 8. Submission
    THRESHOLD = 0.9047083118586069
    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric {final_metric} > {THRESHOLD}. Generating submission..."
        )
        inference(test_loader, model, config.DEVICE)
    else:
        print(f"\nValidation metric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
