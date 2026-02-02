import sys
import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Ensure library imports work by adding current directory to path
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import InkDataset
from library.model import SegFormerB2
from library.train import train_one_epoch, validate, set_seed
from library.inference import run_inference
from library.utils import fbeta_score


def main():
    # --- 1. Configuration Overrides for Fast Baseline ---
    # Limit epochs to ensure execution finishes within the time limit
    Config.NUM_EPOCHS = 5
    # Set submission path explicitly as per requirements
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure output directories exist
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --- 2. Reproducibility & Device Setup ---
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # --- 3. Data Loading ---
    # Load datasets using the provided library class.
    # We use the full dataset (limit=None) as it is small (412 train samples).
    train_ds = InkDataset(mode="train", limit=None, load_cached_data=True)
    val_ds = InkDataset(mode="validation", limit=None, load_cached_data=True)

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

    # --- 4. Model Initialization ---
    model = SegFormerB2().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # --- 5. Training Loop ---
    best_val_score = -float("inf")
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(Config.NUM_EPOCHS):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Save best model
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), checkpoint_path)

    # --- 6. Final Evaluation ---
    # Load the best model weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Compute final metric on the entire hold-out validation set
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # --- 7. Failure Analysis ---
    # Calculate correlation between Error (1 - F0.5) and Input Mean Intensity
    errors = []
    intensities = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Iterate over batch to calculate per-sample statistics
            for i in range(images.size(0)):
                img = images[i]
                lbl = labels[i]
                prob = probs[i]

                # Feature: Mean Intensity of the input image
                intensity = img.mean().item()

                # Error: 1.0 - F0.5 Score for this specific sample
                # We unsqueeze to add batch dimension required by fbeta_score
                score = fbeta_score(
                    prob.unsqueeze(0),
                    lbl.unsqueeze(0),
                    beta=Config.METRIC_BETA,
                    threshold=Config.THRESHOLD,
                )
                error = 1.0 - score.item()

                intensities.append(intensity)
                errors.append(error)

    if len(errors) > 1:
        # Calculate Pearson correlation coefficient
        corr = np.corrcoef(errors, intensities)[0, 1]
        print(f"Correlation between Error (1-F0.5) and Input Mean Intensity: {corr}")

    # --- 8. Conditional Submission ---
    THRESHOLD = 0.597622633
    if final_metric > THRESHOLD:
        run_inference(model, device)
    else:
        # Explicitly skip submission if threshold is not met
        pass


if __name__ == "__main__":
    main()
