import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config, set_seed
from library.dataset import get_dataset
from library.model import Hybrid3D2DUNet
from library.engine import run_training, evaluate
from library.inference import generate_submission


def analyze_failures(model, dataloader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    errors = []
    means = []
    stds = []
    xs = []
    ys = []

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Calculate Mean Absolute Error (MAE) per sample
            # Shape: (B, 1, H, W) -> (B,)
            batch_errors = torch.abs(probs - masks).mean(dim=(1, 2, 3)).cpu().numpy()
            errors.extend(batch_errors)

            # Extract features from the input volume
            # images shape: (B, 1, D, H, W)
            batch_means = images.mean(dim=(1, 2, 3, 4)).cpu().numpy()
            batch_stds = images.std(dim=(1, 2, 3, 4)).cpu().numpy()

            means.extend(batch_means)
            stds.extend(batch_stds)

            # Extract metadata coordinates
            xs.extend(batch["x"].numpy())
            ys.extend(batch["y"].numpy())

    # Create DataFrame for analysis
    df = pd.DataFrame(
        {
            "error": errors,
            "mean_intensity": means,
            "std_intensity": stds,
            "x": xs,
            "y": ys,
        }
    )

    # Calculate correlations
    correlations = df.corr()["error"].drop("error")

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.to_string())


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading
    # We use the full dataset as it is small (455 train, 114 val) and fits within time limits.
    train_dataset = get_dataset("train", load_cached_data=True)
    val_dataset = get_dataset("val", load_cached_data=True)

    if len(train_dataset) == 0:
        print("Training dataset is empty.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # 3. Model Initialization
    model = Hybrid3D2DUNet().to(device)

    # 4. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 5. Training
    # run_training handles the loop, validation, and checkpoint saving
    _ = run_training(model, train_loader, val_loader, optimizer, device)

    # 6. Final Evaluation & Analysis
    # Load the best model saved during training
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))

    model.eval()

    # Calculate Final Validation Metric
    # We pass a standard criterion required by the evaluate function signature
    criterion = nn.BCEWithLogitsLoss()
    _, final_score, _ = evaluate(model, val_loader, criterion, device)

    # Print the required metric
    print(f"Final Validation Metric: {final_score}")

    # Perform Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Submission
    # Generate submission only if metric threshold is met
    if final_score > 0.41758:
        generate_submission(model, device)
    else:
        print(
            f"Validation score {final_score} is not higher than 0.41758. Skipping submission."
        )


if __name__ == "__main__":
    main()
