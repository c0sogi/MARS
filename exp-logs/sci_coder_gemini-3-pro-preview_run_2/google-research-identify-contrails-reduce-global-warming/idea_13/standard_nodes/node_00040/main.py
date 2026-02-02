import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloader, load_metadata
from library.model import MacroContextUNet
from library.trainer import Trainer
from library.inference import predict_and_submit
from library.metrics import GlobalDiceMetric

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Initialize Config with optimized parameters.
    # Increasing epochs to 30 to ensure convergence (Cite solution_lesson_node_00007).
    config = Config(debug=False, epochs=30, batch_size=32)

    # Ensure output directory is set correctly for this run
    config.idea_name = "optimized_solution"
    config.output_dir = os.path.join(config.working_dir, config.idea_name)
    os.makedirs(config.output_dir, exist_ok=True)

    # Set random seeds for reproducibility
    seed_everything(config.seed)

    print(
        f"Configuration: Epochs={config.epochs}, Batch Size={config.batch_size}, Device={config.device}"
    )

    # ==========================================
    # 2. Training
    # ==========================================
    # Initialize Trainer
    trainer = Trainer(config)

    # Run training
    # Patience set to 3 to stop early if no improvement
    trainer.fit(patience=3)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Final Validation and Failure Analysis...")

    # Load the best model weights
    model = MacroContextUNet(config)
    model.to(config.device)
    best_model_path = config.get_model_save_path("best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model not found. Training might have failed.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=config.device))
    model.eval()

    # Get Validation DataLoader
    valid_loader = get_dataloader(config, mode="validation")

    # Initialize Metric
    global_metric = GlobalDiceMetric(threshold=config.threshold)

    # Container for per-sample scores
    sample_scores = []

    # Inference loop on Validation Set
    with torch.no_grad():
        for images, masks in valid_loader:
            images = images.to(config.device)
            masks = masks.to(config.device)

            # Forward pass
            logits = model(images)

            # Update Global Metric
            global_metric.update(logits, masks)

            # --- Per-Sample Analysis ---
            probs = torch.sigmoid(logits)
            preds = (probs > config.threshold).float()

            # Compute Dice per image in the batch
            # Shape: (B, 1, H, W) -> Sum over (1, 2, 3)
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

            # Dice formula: 2*I / (U + eps)
            # Add epsilon to avoid division by zero
            dice = (2.0 * intersection) / (union + 1e-6)

            # Store scores (move to CPU numpy)
            sample_scores.extend(dice.cpu().numpy())

    # Compute and Print Final Metric
    final_metric = global_metric.compute()
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Load validation metadata
    # Note: get_dataloader uses load_metadata internally, so the order is consistent
    # provided shuffle=False (which is default for validation in dataset.py)
    val_meta = load_metadata(config, mode="validation")

    if len(val_meta) == len(sample_scores):
        # Add scores to dataframe
        val_meta["dice"] = sample_scores
        val_meta["error"] = 1.0 - val_meta["dice"]

        # Feature Engineering for Correlation
        # Convert timestamp to hour of day
        val_meta["datetime"] = pd.to_datetime(val_meta["timestamp"], unit="s")
        val_meta["hour"] = val_meta["datetime"].dt.hour

        # Select features to analyze
        features = ["timestamp", "hour", "row_min", "col_min"]

        print("\nFailure Analysis - Correlation with Error (1 - Dice):")
        for feat in features:
            if feat in val_meta.columns:
                corr = val_meta[feat].corr(val_meta["error"])
                print(f"  Correlation between '{feat}' and Error: {corr:.6f}")
    else:
        print(
            f"Warning: Metadata length ({len(val_meta)}) matches scores length ({len(sample_scores)}). Skipping detailed correlation analysis."
        )

    # ==========================================
    # 4. Submission
    # ==========================================
    THRESHOLD = 0.5910660985501295

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # Run inference on test set and generate submission.csv
        predict_and_submit(config)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
