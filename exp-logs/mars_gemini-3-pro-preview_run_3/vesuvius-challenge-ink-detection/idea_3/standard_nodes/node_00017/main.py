import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import InkDataset
import importlib
import library.model

importlib.reload(library.model)
from library.model import PSDN
from library.train import train_model
from library.inference import generate_submission


def calculate_correlation(x, y):
    """
    Calculates Pearson correlation coefficient using numpy.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x = np.array(x)
    y = np.array(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def run_failure_analysis(model, device):
    """
    Runs inference on the validation set to analyze error correlations.
    """
    print("\n--- Failure Analysis ---")

    # Load validation dataset (same seed ensures same subset as training)
    val_dataset = InkDataset(split="val", transform=None, cache_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()

    errors = []
    vol_means = []
    vol_stds = []

    print(f"Analyzing {len(val_dataset)} validation samples...")

    with torch.no_grad():
        for volumes, labels in val_loader:
            volumes = volumes.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(volumes)
            probs = torch.sigmoid(logits)

            # Calculate Mean Absolute Error (MAE) per patch
            # Shape: (B, 1, H, W) -> (B,)
            batch_errors = torch.abs(probs - labels).mean(dim=(1, 2, 3)).cpu().numpy()
            errors.extend(batch_errors)

            # Calculate Input Features per patch
            # Shape: (B, D, H, W) -> (B,)
            batch_means = volumes.mean(dim=(1, 2, 3)).cpu().numpy()
            batch_stds = volumes.std(dim=(1, 2, 3)).cpu().numpy()

            vol_means.extend(batch_means)
            vol_stds.extend(batch_stds)

    # Calculate Correlations
    corr_mean = calculate_correlation(errors, vol_means)
    corr_std = calculate_correlation(errors, vol_stds)

    print(f"Correlation (Error vs Input Mean Intensity): {corr_mean:.10f}")
    print(f"Correlation (Error vs Input Std Deviation): {corr_std:.10f}")


def main():
    # 1. Configuration for Fast Baseline
    Config.set_seed(Config.SEED)

    # Adjust hyperparameters for a fast run within time limits
    Config.NUM_EPOCHS = 5
    Config.STEPS_PER_EPOCH = 200
    Config.VAL_SAMPLE_SIZE = 2000

    # Configure submission path
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    Config.SUBMISSION_FILE = os.path.join(submission_dir, "submission.csv")

    print("Configuration set for fast baseline.")
    print(f"Epochs: {Config.NUM_EPOCHS}, Steps: {Config.STEPS_PER_EPOCH}")

    # 2. Train Model
    # train_model returns (best_val_score, best_threshold)
    best_score, best_threshold = train_model(load_cached_data=True)

    # 3. Print Required Metric
    # Must print full precision
    print(f"Final Validation Metric: {best_score}")

    # 4. Failure Analysis
    device = torch.device(Config.DEVICE)
    model = PSDN().to(device)

    # Load best model weights
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        run_failure_analysis(model, device)
    else:
        print("Warning: Best model not found for failure analysis.")

    # 5. Generate Submission
    # Condition: Metric > 0.39266693592071533
    threshold_cutoff = 0.39266693592071533

    if best_score > threshold_cutoff:
        print(f"\nValidation score {best_score} exceeds {threshold_cutoff}.")
        print("Generating submission file...")
        generate_submission(threshold=best_threshold, load_cached_data=True)
    else:
        print(f"\nValidation score {best_score} does not exceed {threshold_cutoff}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
