import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported from the current directory
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import CactusResNet, train_model, predict_and_submit
from library.utils import load_checkpoint, calculate_auc


def main():
    # 1. Setup System
    Config.create_directories()
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing task on device: {device}")

    # 2. Prepare Data
    # Load data with caching enabled for efficiency
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Model
    model = CactusResNet().to(device)

    # 4. Train Model
    # Using the defined number of epochs (50) which is feasible within the time limit for this dataset size
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 5. Validation Assessment
    best_model_path = Config.OUTPUT_MODEL_PATH
    if not os.path.exists(best_model_path):
        print(f"Error: Checkpoint not found at {best_model_path}")
        return

    print(f"Loading best model weights from {best_model_path}...")
    checkpoint = load_checkpoint(best_model_path, model, device=device)
    model.eval()

    val_probs = []
    val_targets = []

    # Storage for image stats for failure analysis
    val_stats = {
        "brightness": [],
        "contrast": [],
        "red_mean": [],
        "green_mean": [],
        "blue_mean": [],
    }

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)

            val_probs.extend(probs.cpu().numpy())
            val_targets.extend(labels.numpy())

            # Extract features for failure analysis
            # Images are (B, 3, 32, 32) tensors
            # Brightness: Mean over all pixels (channels, height, width)
            b_batch = images.mean(dim=[1, 2, 3]).cpu().numpy()
            val_stats["brightness"].extend(b_batch)

            # Contrast: Standard deviation over all pixels
            c_batch = images.std(dim=[1, 2, 3]).cpu().numpy()
            val_stats["contrast"].extend(c_batch)

            # Channel Means: Mean over height and width
            means = images.mean(dim=[2, 3]).cpu().numpy()  # Shape (B, 3)
            val_stats["red_mean"].extend(means[:, 0])
            val_stats["green_mean"].extend(means[:, 1])
            val_stats["blue_mean"].extend(means[:, 2])

    # Calculate Metric
    val_auc = calculate_auc(val_targets, val_probs)

    # Print Metric in required format
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    val_probs_arr = np.array(val_probs)
    val_targets_arr = np.array(val_targets)

    # Calculate error magnitude
    errors = np.abs(val_targets_arr - val_probs_arr)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(val_stats)
    df_analysis["error"] = errors

    print("Correlation between Error Magnitude and Input Features:")
    # Calculate Pearson correlation
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    # 7. Submission
    THRESHOLD = 0.9997903583412834

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader, device)
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
