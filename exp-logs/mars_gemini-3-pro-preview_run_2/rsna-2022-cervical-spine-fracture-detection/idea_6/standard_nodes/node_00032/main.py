import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure library is in path
sys.path.append("./library")

from library.config import Config
from library.utils import seed_everything
from library.dataset import CervicalSpineDataset
from library.model import CervicalSpineTransformer
from library.loss import WeightedMultiLabelLogLoss
from library.engine import train_one_epoch, validate, predict_test_set


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Fast Baseline Configuration
    EPOCHS = 5

    print(f"Initializing Fast Baseline Run (Epochs: {EPOCHS})...")

    # 2. Data Loading
    # We use debug=False to use the full provided dataset (161 train, 41 val)
    # load_cached_data=True is default in Dataset class
    train_dataset = CervicalSpineDataset(split="train", debug=False)
    val_dataset = CervicalSpineDataset(split="val", debug=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Training Components
    model = CervicalSpineTransformer().to(device)

    # Training Loss: Includes pos_weight for stability/sensitivity
    train_criterion = WeightedMultiLabelLogLoss(pos_weight=Config.POS_WEIGHT).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    total_steps = EPOCHS * len(train_loader) // Config.ACCUMULATION_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=Config.ETA_MIN
    )

    scaler = torch.cuda.amp.GradScaler()

    # 4. Training Loop
    best_val_loss = float("inf")

    print("Starting Training...")
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, train_criterion, device, scaler, scheduler
        )

        # Validate (using training criterion for checkpointing)
        val_loss = validate(model, val_loader, train_criterion, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss (Proxy): {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)

    # 5. Final Validation Metric & Failure Analysis
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
    model.eval()

    val_probs = []
    val_targets = []
    slice_counts = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)

            val_probs.append(probs.cpu())
            val_targets.append(targets.cpu())

            # Collect metadata for failure analysis (slice count)
            start_idx = i * Config.BATCH_SIZE
            end_idx = start_idx + images.size(0)
            batch_df = val_dataset.df.iloc[start_idx:end_idx]
            batch_counts = batch_df["slice_files"].apply(len).values
            slice_counts.extend(batch_counts)

    val_probs = torch.cat(val_probs)
    val_targets = torch.cat(val_targets)
    slice_counts = np.array(slice_counts)

    # Calculate Metric according to Task Description
    # Formula: L_ij = -w_j * [y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij)]
    # Weights: 1 for C1-C7, 7 for Patient Overall
    class_weights = torch.tensor([1.0] * 7 + [7.0])

    # Clamp for numerical stability
    eps = 1e-7
    val_probs = torch.clamp(val_probs, eps, 1 - eps)

    # Compute Log Loss per element
    log_loss = -(
        val_targets * torch.log(val_probs)
        + (1 - val_targets) * torch.log(1 - val_probs)
    )

    # Apply class weights
    weighted_log_loss = log_loss * class_weights.unsqueeze(0)

    # Average across all rows (and columns, implicitly, to get a single scalar metric)
    final_metric = weighted_log_loss.mean().item()

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # Error magnitude per sample (average weighted loss across the 8 targets)
    sample_errors = weighted_log_loss.mean(dim=1).numpy()

    if len(slice_counts) > 1:
        correlation = np.corrcoef(sample_errors, slice_counts)[0, 1]
        print(f"Correlation between Error and Slice Count: {correlation:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 7. Submission
    THRESHOLD = 0.15364714496434773

    if final_metric < THRESHOLD:
        print("Metric below threshold. Generating submission...")

        test_dataset = CervicalSpineDataset(split="test", debug=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_df = predict_test_set(model, test_loader, device)

        # Save to ./submission/submission.csv as requested
        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        sub_path = os.path.join(sub_dir, "submission.csv")

        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
