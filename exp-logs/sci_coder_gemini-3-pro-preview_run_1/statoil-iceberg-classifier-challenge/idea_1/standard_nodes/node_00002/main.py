import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import get_dataloaders
from library.model import train_model, generate_submission, set_seeds


def main():
    # 1. Configuration and Setup
    config = Config()
    # Use config default (75) or set explicitly here if needed
    # config.NUM_EPOCHS = 75

    # Set seeds for reproducibility
    set_seeds(config.SEED)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Load cached data for efficiency
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Training
    # train_model handles the training loop, early stopping, and returns the best model
    model = train_model(train_loader, val_loader, config)

    # 4. Validation Assessment
    model.eval()
    criterion = nn.BCELoss()
    val_loss = 0.0

    # Containers for failure analysis
    predictions = []
    targets = []
    inc_angles = []
    img_means = []
    img_stds = []

    with torch.no_grad():
        for imgs, angles, labels in val_loader:
            # Move data to device
            imgs = imgs.to(device)
            angles_dev = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(imgs, angles_dev)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)

            # Collect data for failure analysis
            preds_np = outputs.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            angles_np = angles.numpy()

            predictions.extend(preds_np)
            targets.extend(labels_np)
            inc_angles.extend(angles_np)

            # Calculate simple image statistics (mean and std intensity)
            # imgs shape: (Batch, Channels, Height, Width)
            imgs_cpu = imgs.cpu().numpy()
            imgs_flat = imgs_cpu.reshape(imgs_cpu.shape[0], -1)

            img_means.extend(np.mean(imgs_flat, axis=1))
            img_stds.extend(np.std(imgs_flat, axis=1))

    # Compute final metric
    final_metric = val_loss / len(val_loader.dataset)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    df_analysis = pd.DataFrame(
        {
            "prediction": predictions,
            "target": targets,
            "inc_angle": inc_angles,
            "img_mean": img_means,
            "img_std": img_stds,
        }
    )

    # Calculate Error Magnitude
    df_analysis["error_magnitude"] = np.abs(
        df_analysis["prediction"] - df_analysis["target"]
    )

    # Calculate Correlations between Error Magnitude and Input Features
    features = ["inc_angle", "img_mean", "img_std"]
    correlations = (
        df_analysis[features + ["error_magnitude"]]
        .corr()["error_magnitude"]
        .drop("error_magnitude")
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission
    # Generate predictions for the test set and save to CSV
    if final_metric < 0.2733:
        print(
            f"Validation metric {final_metric:.4f} improved over baseline 0.2733. Generating submission..."
        )
        generate_submission(model, test_loader, config)
    else:
        print(
            f"Validation metric {final_metric:.4f} did not improve over baseline 0.2733. Skipping submission."
        )


if __name__ == "__main__":
    main()
