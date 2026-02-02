import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import PathologyDataset, get_transforms
from library.model import get_model
from library.trainer import Trainer
from library.inference import run_inference


def main():
    # --- 1. Configuration & Setup ---
    # Set seeds for reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # --- 2. Data Loading ---
    print("Loading datasets...")
    # Train loader
    train_dataset = PathologyDataset(mode="train", transform=get_transforms("train"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation loader
    val_dataset = PathologyDataset(mode="val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Model & Optimizer ---
    print("Initializing model...")
    model = get_model(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS
    )

    # --- 4. Training ---
    print("Starting training...")
    trainer = Trainer(
        model, optimizer, train_loader, val_loader, device, scheduler=scheduler
    )
    trainer.fit()

    # --- 5. Final Validation & Metric ---
    print("Performing final validation...")
    # Load the best model saved during training
    if os.path.exists(Config.MODEL_CHECKPOINT):
        model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))
    else:
        print("Warning: Checkpoint not found, using current weights.")

    model.eval()

    val_targets = []
    val_probs = []

    # Store stats for failure analysis
    # We collect these on the fly to avoid iterating twice
    meta_stats = {
        "brightness": [],
        "contrast": [],
        "red_mean": [],
        "green_mean": [],
        "blue_mean": [],
    }

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets_np = targets.numpy()

            # Inference
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            val_targets.extend(targets_np)
            val_probs.extend(probs)

            # --- Feature Extraction for Failure Analysis ---
            # Inputs are (B, 3, H, W) and normalized.
            # We use normalized stats as proxies for image characteristics.

            # Mean per image per channel
            batch_means = inputs.mean(dim=(2, 3)).cpu().numpy()  # (B, 3)
            # Std per image (proxy for contrast)
            batch_stds = inputs.std(dim=(1, 2, 3)).cpu().numpy()  # (B, )
            # Brightness (avg of channels)
            batch_brightness = batch_means.mean(axis=1)  # (B, )

            meta_stats["red_mean"].extend(batch_means[:, 0])
            meta_stats["green_mean"].extend(batch_means[:, 1])
            meta_stats["blue_mean"].extend(batch_means[:, 2])
            meta_stats["brightness"].extend(batch_brightness)
            meta_stats["contrast"].extend(batch_stds)

    val_targets = np.array(val_targets)
    val_probs = np.array(val_probs)

    # Compute and Print Final Metric
    final_auc = roc_auc_score(val_targets, val_probs)
    print(f"Final Validation Metric: {final_auc}")

    # --- 6. Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_probs)

    print("Correlation between Error Magnitude and Input Features:")
    for feature, values in meta_stats.items():
        values = np.array(values)
        if len(values) != len(errors):
            print(f"Warning: Length mismatch for {feature}")
            continue

        # Compute Pearson correlation
        corr, p_val = pearsonr(errors, values)
        print(f"  {feature}: Correlation = {corr:.4f} (p-value = {p_val:.4g})")

    # --- 7. Submission ---
    THRESHOLD = 0.982754282933193

    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        run_inference(
            checkpoint_path=Config.MODEL_CHECKPOINT,
            output_path=Config.SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            device=device,
        )
    else:
        print(f"\nMetric ({final_auc}) <= Threshold ({THRESHOLD}). Submission skipped.")


if __name__ == "__main__":
    main()
