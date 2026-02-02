import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_data, SIIMDataset, get_transforms
from library.model import StochasticResNet34UNet
from library.engine import train_model, predict, evaluate


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    seed_everything(Config.SEED)

    # Adjust Configuration for Fast Baseline within Time Limit
    # Estimating ~10 mins for data processing (if cache missing) and ~1 min/epoch training on A100.
    # 15 Epochs provides a good balance between convergence and runtime safety.
    Config.EPOCHS = 15

    # Ensure verbose output for logging
    print(f"=== Configuration ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # =========================================================================
    # 2. Training Pipeline
    # =========================================================================
    print("\n=== Starting Training Pipeline ===")
    # train_model() handles:
    # - Data loading (and caching)
    # - Model initialization
    # - Training loop with mixed precision
    # - Validation monitoring
    # - Saving the best model to Config.BEST_MODEL_PATH
    train_model()

    # =========================================================================
    # 3. Validation Assessment
    # =========================================================================
    print("\n=== Starting Validation Assessment ===")

    # Check if model was saved successfully
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Best model not found at {Config.BEST_MODEL_PATH}")
        sys.exit(1)

    # Load the best model
    device = Config.DEVICE
    model = StochasticResNet34UNet().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Load Validation Data
    # We explicitly load validation data here to perform the final metric calculation and failure analysis
    val_data = load_data(Config.VAL_METADATA, "val", load_cached_data=True)
    val_dataset = SIIMDataset(val_data, "val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Execute Evaluation
    # evaluate() returns: average_loss, image_map, study_accuracy
    val_loss, val_map, val_acc = evaluate(model, val_loader, device)

    # Calculate Final Metric
    # The competition metric is a composite. We use the same metric used for checkpointing:
    # (Image mAP + Study Accuracy) / 2
    final_metric = (val_map + val_acc) / 2.0

    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Image mAP: {val_map:.6f}")
    print(f"Study Accuracy: {val_acc:.6f}")
    # Print the required metric string
    print(f"Final Validation Metric: {final_metric:.14f}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n=== Performing Failure Analysis ===")

    # We will calculate the correlation between Error Magnitude (Total Loss) and Opacity Area.
    criterion_study = nn.CrossEntropyLoss(reduction="none")
    criterion_mask = nn.BCEWithLogitsLoss(reduction="none")

    errors = []
    opacity_areas = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            study_logits, mask_logits = model(images)

            # 1. Calculate Error Magnitude (Loss) per sample
            # Study Loss
            study_targets = torch.argmax(labels, dim=1)
            loss_s = criterion_study(study_logits, study_targets)

            # Mask Loss (BCE)
            # mask_logits: (B, 1, H, W), masks: (B, 1, H, W)
            # We average over spatial dimensions to get (B,)
            loss_m = criterion_mask(mask_logits, masks).mean(dim=(1, 2, 3))

            # Weighted Total Loss
            total_loss = (Config.STUDY_LOSS_WEIGHT * loss_s) + (
                Config.IMAGE_LOSS_WEIGHT * loss_m
            )
            errors.extend(total_loss.cpu().numpy())

            # 2. Extract Features (Opacity Area)
            # Sum of mask pixels indicates the size of the opacity
            # masks is (B, 1, H, W)
            areas = masks.sum(dim=(1, 2, 3)).cpu().numpy()
            opacity_areas.extend(areas)

    errors = np.array(errors)
    opacity_areas = np.array(opacity_areas)

    # Calculate Pearson Correlation
    if len(errors) > 1 and np.std(opacity_areas) > 0:
        correlation = np.corrcoef(errors, opacity_areas)[0, 1]
        print(f"Correlation (Error Magnitude vs Opacity Area): {correlation:.6f}")
    else:
        print("Correlation could not be calculated due to insufficient variance.")

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    print("\n=== Submission Generation ===")
    THRESHOLD = 0.49944536565378

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # predict() handles test data loading, TTA inference, and CSV generation
        predict()
    else:
        print(
            f"Metric ({final_metric:.6f}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
