import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, get_test_dataloader
from library.model import DDSRNet, predict
from library.train import run_training


def calculate_correlation(x, y):
    """Calculates Pearson correlation coefficient using numpy."""
    if len(x) != len(y):
        raise ValueError("Inputs must have same length")

    n = len(x)
    if n < 2:
        return 0.0

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate covariance and variances
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2)) * np.sqrt(
        np.sum((y - mean_y) ** 2)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def main():
    # 1. Setup and Configuration
    # Limit epochs to ensure quick execution while maintaining performance
    Config.EPOCHS = 30
    Config.setup()
    seed_everything(Config.SEED)

    # 2. Training
    # run_training handles the loop, validation, and saves 'best_model.pth'
    best_metric = run_training(patience=10)

    # Required Output
    print(f"Final Validation Metric: {best_metric}")

    # 3. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")

    device = torch.device(Config.DEVICE)
    model = DDSRNet().to(device)

    # Load best model for analysis
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found. Skipping failure analysis.")
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        _, val_loader = get_dataloaders()

        all_errors = []
        all_features = []

        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                tabular = batch["tabular"].to(device)
                raw_fvc = batch["raw_fvc"].numpy()  # Ground truth in ml

                mu, sigma = model(image, tabular)

                # Inverse transform predictions to calculate real-world error
                mu_np = mu.cpu().numpy().flatten()
                mu_unscaled = mu_np * Config.TARGET_STD + Config.TARGET_MEAN

                # Calculate absolute error
                batch_errors = np.abs(raw_fvc - mu_unscaled)
                all_errors.extend(batch_errors)

                # Store features: [BaseFVC_norm, Age_norm, Sex, Smoke, RelWeek_scaled]
                all_features.extend(tabular.cpu().numpy())

        all_errors = np.array(all_errors)
        all_features = np.array(all_features)

        feature_names = ["BaseFVC_norm", "Age_norm", "Sex", "Smoke", "RelWeek_scaled"]

        print("Correlation between Absolute Error and Input Features:")
        for i, name in enumerate(feature_names):
            feat_values = all_features[:, i]
            corr = calculate_correlation(all_errors, feat_values)
            print(f"  {name}: {corr:.4f}")

    # 4. Submission
    # Threshold defined in task description
    SUBMISSION_THRESHOLD = -6.573619738753321

    # Metric values are negative, higher is better (e.g., -6.5 > -6.6)
    if best_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric {best_metric} exceeds threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        test_loader = get_test_dataloader()
        # predict function loads the best model internally and saves to Config.SUBMISSION_PATH
        predict(test_loader)
    else:
        print(
            f"\nValidation metric {best_metric} does not exceed threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
