import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataset
from library.model import DualStreamModel
from library.engine import train_model, generate_submission


def run():
    # 1. Configuration and Setup
    # Override defaults for a fast baseline execution as per requirements
    Config.setup()
    Config.EPOCHS = 5  # Reduced from 10 to ensure completion within time limits

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device selected: {device}")

    # 2. Data Loading
    print("Initializing datasets and dataloaders...")
    # Using 'debug=Config.DEBUG' which defaults to False, ensuring full dataset usage unless changed
    train_dataset = get_dataset("train", debug=Config.DEBUG)
    val_dataset = get_dataset("val", debug=Config.DEBUG)

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

    # 3. Model Training
    print("Starting training loop...")
    # train_model handles the training loop, validation per epoch, and saving the best model
    train_model(
        train_loader,
        val_loader,
        device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 4. Final Validation & Failure Analysis
    print("Loading best model for final validation and failure analysis...")

    # Load the best saved model
    model = DualStreamModel(pretrained=False)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Metrics containers
    correct_count = 0
    total_count = 0
    all_losses = []
    all_means = []
    all_stds = []

    # Loss for analysis (reduction='none' to get per-sample loss)
    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate per-sample loss for failure analysis
            losses = criterion(outputs, labels)
            all_losses.append(losses.cpu().numpy())

            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            correct_count += (preds == labels).sum().item()
            total_count += images.size(0)

            # Extract simple input features for correlation analysis
            # images is (B, C, H, W). We take mean/std across C, H, W for a scalar per image
            # Note: images are normalized, but this still serves as a proxy for signal properties
            batch_means = images.mean(dim=(1, 2, 3)).cpu().numpy()
            batch_stds = images.std(dim=(1, 2, 3)).cpu().numpy()

            all_means.append(batch_means)
            all_stds.append(batch_stds)

    # Compute and print final metric
    final_metric = correct_count / total_count
    print(f"Final Validation Metric: {final_metric}")

    # Compute correlations for failure analysis
    all_losses = np.concatenate(all_losses)
    all_means = np.concatenate(all_means)
    all_stds = np.concatenate(all_stds)

    corr_mean = np.corrcoef(all_losses, all_means)[0, 1]
    corr_std = np.corrcoef(all_losses, all_stds)[0, 1]

    print(
        "Failure Analysis - Correlation between Error Magnitude (Loss) and Input Features:"
    )
    print(f"  Correlation with Mean Pixel Intensity: {corr_mean}")
    print(f"  Correlation with Std Pixel Intensity:  {corr_std}")

    # 5. Submission Generation
    # Strict threshold check
    THRESHOLD = 0.9022696929238986

    if final_metric > THRESHOLD:
        print(
            f"Metric {final_metric} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        test_dataset = get_dataset("test", debug=Config.DEBUG)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # generate_submission handles TTA and file saving
        generate_submission(
            test_loader,
            device,
            model_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"Metric {final_metric} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
