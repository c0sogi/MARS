import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_rmse
from library.model import RepCResUNetSR
from library.train import train_model
from library.data_loader import get_dataloaders
from library.inference import generate_submission


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Define fast baseline parameters
    # We use fewer epochs to ensure execution within the time limit.
    # 10 epochs with the default patch settings is sufficient for a baseline
    # and runs efficiently on the provided A100 GPU.
    FAST_EPOCHS = 10

    print(f"Starting execution with {FAST_EPOCHS} epochs...")

    # 2. Train the Model
    # We pass the epochs explicitly to override the default Config.EPOCHS
    best_rmse_from_train = train_model(epochs=FAST_EPOCHS)

    # 3. Validation and Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")

    device = Config.DEVICE
    model = RepCResUNetSR().to(device)

    # Load the best model saved during training
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )

    state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # Switch to deploy mode (fuses RepBlocks) for efficient inference
    model.eval()
    model.switch_to_deploy()

    # Get validation loader
    # train_model uses batch_size=Config.BATCH_SIZE, we stick to defaults for val (batch_size=1)
    loaders = get_dataloaders()
    val_loader = loaders["val"]

    val_errors = []
    input_means = []
    input_stds = []

    total_rmse = 0.0
    count = 0

    # Inference loop without gradients
    with torch.no_grad():
        for noisy_imgs, clean_imgs, img_ids in val_loader:
            noisy_imgs = noisy_imgs.to(device)
            # clean_imgs is usually on CPU from loader, calculate_rmse handles it

            # Predict
            preds = model(noisy_imgs)

            # Calculate RMSE for this image
            rmse = calculate_rmse(clean_imgs, preds)

            total_rmse += rmse
            count += 1

            # Collect stats for failure analysis
            # Input features: Mean intensity and Standard Deviation of the noisy image
            noisy_flat = noisy_imgs.cpu().numpy().flatten()
            mean_intensity = np.mean(noisy_flat)
            std_intensity = np.std(noisy_flat)

            val_errors.append(rmse)
            input_means.append(mean_intensity)
            input_stds.append(std_intensity)

    # Compute Final Metric
    final_metric = total_rmse / count

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    # Check if we have enough data points
    if len(val_errors) > 1:
        corr_mean, _ = pearsonr(val_errors, input_means)
        corr_std, _ = pearsonr(val_errors, input_stds)

        print(f"Correlation (Error vs Input Mean Intensity): {corr_mean}")
        print(f"Correlation (Error vs Input Std Deviation): {corr_std}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 4. Submission Generation
    # Threshold defined in task
    THRESHOLD = 0.0076658159

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(
            checkpoint_path=Config.BEST_MODEL_PATH,
            output_path=Config.SUBMISSION_PATH,
            test_csv_path=Config.TEST_CSV,
            device=device,
        )
    else:
        print(
            f"\nValidation metric {final_metric} is NOT below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
