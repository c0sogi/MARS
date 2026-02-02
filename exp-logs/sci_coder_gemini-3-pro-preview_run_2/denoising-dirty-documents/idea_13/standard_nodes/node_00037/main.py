import sys
import os
import torch
import numpy as np
from scipy.stats import pearsonr
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(os.getcwd())

# Import from library
from library.config import Config
from library.utils import seed_everything
from library.train import run_training
from library.inference import create_submission
from library.dataset import DenoisingDataset
from library.model import CoRes2NetUNet


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Adjust Config for Fast Baseline execution
    # We reduce epochs to ensure completion within the time limit while maintaining
    # high sampling density for performance.
    Config.NUM_EPOCHS = 30

    print(f"Starting Fast Baseline Run (Epochs: {Config.NUM_EPOCHS})...")

    # 2. Training
    # run_training handles data loading, model init, training loop, and validation
    best_rmse = run_training(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Required Output Format
    print(f"Final Validation Metric: {best_rmse}")

    # 3. Failure Analysis
    print("\n--- Failure Analysis ---")
    device = Config.DEVICE

    # Load the best model for analysis
    model = CoRes2NetUNet().to(device)
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        checkpoint = torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: Checkpoint not found. Skipping analysis with trained weights.")

    model.eval()

    # Load validation data
    val_dataset = DenoisingDataset(mode="val", load_cached_data=True)

    all_errors = []
    all_clean_vals = []
    all_noisy_vals = []

    # Iterate through validation set to collect stats
    with torch.no_grad():
        for i in range(len(val_dataset)):
            noisy_t, clean_t, _ = val_dataset[i]

            # Move to GPU
            noisy_gpu = noisy_t.unsqueeze(0).to(device)

            # Predict Noise
            pred_noise = model(noisy_gpu)

            # Reconstruct Clean
            pred_clean = noisy_gpu - pred_noise
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

            # Move to CPU/Numpy for stats
            pred_clean_np = pred_clean.cpu().numpy().flatten()
            clean_np = clean_t.numpy().flatten()
            noisy_np = noisy_t.numpy().flatten()

            # Calculate Absolute Error
            error = np.abs(pred_clean_np - clean_np)

            all_errors.append(error)
            all_clean_vals.append(clean_np)
            all_noisy_vals.append(noisy_np)

    # Concatenate all pixels
    flat_errors = np.concatenate(all_errors)
    flat_clean = np.concatenate(all_clean_vals)
    flat_noisy = np.concatenate(all_noisy_vals)

    # Calculate Correlations
    # We use a subset if the array is massive to speed up calculation,
    # but for ~10M pixels (23 images), full calculation is fine on modern CPUs.
    corr_clean, _ = pearsonr(flat_errors, flat_clean)
    corr_noisy, _ = pearsonr(flat_errors, flat_noisy)

    print(f"Correlation (Error vs Clean Intensity): {corr_clean:.4f}")
    print(f"Correlation (Error vs Noisy Intensity): {corr_noisy:.4f}")

    # 4. Submission Generation
    # Only generate if performance is sufficient
    THRESHOLD = 0.0076658159

    if best_rmse < THRESHOLD:
        print(f"\nValidation metric {best_rmse} is better than threshold {THRESHOLD}.")
        create_submission()
    else:
        print(
            f"\nValidation metric {best_rmse} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
