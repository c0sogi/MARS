import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.dataset import SETIDataset
from library.model import SiameseModel
from library.engine import run_training, validate
from library.utils import seed_everything, apply_tta


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model failure modes by correlating prediction error with input signal statistics.
    """
    print("\n=== Performing Failure Analysis ===")
    model.eval()

    stats_list = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Failure Analysis", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            # Get predictions
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)

            # Calculate Error Magnitude
            errors = torch.abs(targets - probs)

            # Extract Image Stats (B, 6, H, W)
            # On-Target: Channels 0, 2, 4
            # Off-Target: Channels 1, 3, 5
            on_target = images[:, [0, 2, 4], :, :]
            off_target = images[:, [1, 3, 5], :, :]

            # Flatten spatial/channel dims for simple stats per sample
            B = images.size(0)
            on_flat = on_target.reshape(B, -1)
            off_flat = off_target.reshape(B, -1)

            # Compute statistics
            mean_on = torch.mean(on_flat, dim=1)
            std_on = torch.std(on_flat, dim=1)
            max_on = torch.max(on_flat, dim=1).values

            mean_off = torch.mean(off_flat, dim=1)
            std_off = torch.std(off_flat, dim=1)
            max_off = torch.max(off_flat, dim=1).values

            mean_diff = mean_on - mean_off

            # Collect data
            batch_stats = pd.DataFrame(
                {
                    "error": errors.cpu().numpy(),
                    "mean_on": mean_on.cpu().numpy(),
                    "std_on": std_on.cpu().numpy(),
                    "max_on": max_on.cpu().numpy(),
                    "mean_off": mean_off.cpu().numpy(),
                    "std_off": std_off.cpu().numpy(),
                    "max_off": max_off.cpu().numpy(),
                    "mean_diff": mean_diff.cpu().numpy(),
                    "target": targets.cpu().numpy(),
                }
            )
            stats_list.append(batch_stats)

    df_stats = pd.concat(stats_list, ignore_index=True)

    # Calculate Correlations with Error
    # We drop 'error' (self-correlation) and 'target'
    correlations = (
        df_stats.corr()["error"].drop(["error", "target"]).sort_values(ascending=False)
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    return correlations


def main():
    # --- 1. Configuration Override ---
    # Override submission path to match Task Description
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Model Path: {Config.MODEL_PATH}")
    print(f"  Submission Path: {Config.SUBMISSION_PATH}")

    seed_everything(Config.SEED)

    # --- 2. Data Loading ---
    print("Loading Datasets...")
    train_dataset = SETIDataset(mode="train")
    val_dataset = SETIDataset(mode="val")

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

    # --- 3. Model Setup ---
    print("Initializing Model...")
    model = SiameseModel(model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED)
    model = model.to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Adjust scheduler T_max to match the reduced epoch count
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    criterion = nn.BCEWithLogitsLoss()

    # --- 4. Training ---
    print("Starting Training...")
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS,
        save_path=Config.MODEL_PATH,
        patience=3,
    )

    # --- 5. Final Validation & Metrics ---
    print("Loading Best Model for Validation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))

    _, final_auc = validate(model, val_loader, criterion, Config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # --- 6. Failure Analysis ---
    perform_failure_analysis(model, val_loader, Config.DEVICE)

    # --- 7. Submission ---
    threshold = 0.7930069652683209

    if final_auc > threshold:
        print(
            f"Validation AUC ({final_auc}) > Threshold ({threshold}). Generating Submission..."
        )

        test_dataset = SETIDataset(mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model.eval()
        all_preds = []
        ids = test_dataset.df["id"].tolist()

        # Inference with TTA
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc="Test Inference"):
                # apply_tta handles moving to device and sigmoid
                avg_probs = apply_tta(model, images, Config.DEVICE)
                all_preds.extend(avg_probs.cpu().numpy())

        submission = pd.DataFrame({"id": ids, "target": all_preds})

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation AUC ({final_auc}) <= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
