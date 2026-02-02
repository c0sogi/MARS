import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import set_seed
from library.dataset import ContrailDataset
from library.training import ContrailTrainer

# --- Configuration Overrides ---
# Increasing epochs to 30 for better convergence (Cite {solution_lesson_node_00009})
Config.EPOCHS = 30
Config.T_MAX = 30
Config.AVERAGE_START_EPOCH = 15
Config.DEBUG = False


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Initialize Trainer
    trainer = ContrailTrainer()

    # --- Data Loading ---
    # Load full datasets
    train_dataset = ContrailDataset(split="train", load_cached_data=True)
    val_dataset = ContrailDataset(split="validation", load_cached_data=True)

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

    print(f"Starting training for {Config.EPOCHS} epochs on device {Config.DEVICE}...")

    # --- Training Loop ---
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = trainer.train_one_epoch(train_loader)

        # Validate (Standard validation for checkpointing)
        val_loss, val_dice = trainer.validate(val_loader)

        # Update Scheduler
        trainer.scheduler.step()

        # Save Checkpoint
        trainer.save_checkpoint(epoch, val_dice)

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

    # --- Weight Averaging ---
    best_model_path = trainer.average_weights()

    # --- Final Validation & Failure Analysis ---
    print("\nRunning Final Validation and Failure Analysis...")

    # Load the best averaged model
    state_dict = torch.load(best_model_path, map_location=trainer.device)
    trainer.model.load_state_dict(state_dict)
    trainer.model.eval()

    total_inter = 0.0
    total_union = 0.0
    sample_dices = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(trainer.device)
            masks = masks.to(trainer.device)

            # Forward pass
            outputs = trainer.model(images)
            preds = torch.sigmoid(outputs)
            preds_bin = (preds > 0.5).float()

            # --- Global Dice Calculation ---
            # Flatten batch for intersection/union accumulation
            p_flat = preds_bin.view(-1)
            m_flat = masks.view(-1)

            inter = (p_flat * m_flat).sum().item()
            union = p_flat.sum().item() + m_flat.sum().item()

            total_inter += inter
            total_union += union

            # --- Per-Sample Dice for Failure Analysis ---
            # Iterate through batch
            for j in range(images.size(0)):
                p_s = preds_bin[j].view(-1)
                m_s = masks[j].view(-1)

                s_inter = (p_s * m_s).sum().item()
                s_union = p_s.sum().item() + m_s.sum().item()

                # Smooth dice for sample to avoid div by zero
                s_dice = (2.0 * s_inter) / (s_union + 1e-6)
                sample_dices.append(s_dice)

    # Compute Global Dice
    global_dice = (2.0 * total_inter) / (total_union + 1e-6)

    # Print required metric format
    print(f"Final Validation Metric: {global_dice}")

    # --- Failure Analysis ---
    # Load metadata to correlate with errors
    meta = val_dataset.metadata.copy()

    # Align metadata with predictions (assuming loader order matches metadata)
    # Truncate metadata if loader dropped last batch (unlikely for val but safe to check)
    if len(meta) > len(sample_dices):
        meta = meta.iloc[: len(sample_dices)].copy()

    meta["dice"] = sample_dices
    meta["error"] = 1.0 - meta["dice"]

    # Ensure features are numeric
    for col in ["timestamp", "row_min", "col_min"]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")

    # Calculate correlations
    correlations = meta[["error", "timestamp", "row_min", "col_min"]].corr()["error"]

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    print(f"Timestamp: {correlations.get('timestamp', 0):.4f}")
    print(f"Row Min (Latitude): {correlations.get('row_min', 0):.4f}")
    print(f"Col Min (Longitude): {correlations.get('col_min', 0):.4f}")

    # --- Submission ---
    THRESHOLD = 0.6190814624168179

    if global_dice > THRESHOLD:
        print(
            f"\nMetric ({global_dice:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.inference(best_model_path)
    else:
        print(
            f"\nMetric ({global_dice:.6f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
