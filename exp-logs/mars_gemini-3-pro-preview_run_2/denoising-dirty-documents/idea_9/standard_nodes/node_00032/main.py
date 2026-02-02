import os
import sys
import numpy as np
import torch
import warnings
import pandas as pd
from torch.utils.data import DataLoader

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import set_seed, calculate_rmse
from library.trainer import Trainer
from library.dataset import DenoisingDataset


def main():
    # 1. Configuration
    # Initialize config with reduced epochs for a fast baseline execution
    # We use 10 epochs to ensure convergence within the time limit while demonstrating learning
    config = Config(debug=False, num_epochs=10)

    # Further reduce computational load for "fast baseline" requirement
    # Reduce patches per image from default 100 to 20 to speed up epoch time
    config.PATCHES_PER_IMAGE = 20

    # Ensure reproducibility
    set_seed(config.SEED)

    # 2. Training
    # Initialize Trainer with the modified config
    trainer = Trainer(config)

    # Fit the model
    # The trainer handles the training loop, checkpointing, and logging
    trainer.fit()

    # 3. Validation & Failure Analysis
    print("Starting Failure Analysis...")

    # Load the best checkpoint found during training for the final analysis
    if os.path.exists(config.CHECKPOINT_PATH):
        trainer.model.load_state_dict(
            torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        )

    trainer.model.eval()

    # Create validation loader
    # load_cached_data=True ensures we use pre-processed numpy files if available
    val_dataset = DenoisingDataset("val", config, load_cached_data=True)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS
    )

    rmse_list = []

    # Containers for failure analysis correlations
    flat_errors = []
    flat_noisy = []
    flat_clean = []

    with torch.no_grad():
        for noisy, clean, img_id in val_loader:
            noisy = noisy.to(config.DEVICE)
            clean_target = clean.numpy()

            # Predict using tiled inference (handles TTA if enabled in config)
            pred_noise = trainer.predict_image_tiled(noisy)

            # Reconstruct clean image: Clean = Noisy - Noise
            pred_clean = noisy - pred_noise
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

            pred_clean_np = pred_clean.cpu().numpy()

            # Calculate RMSE for this image
            img_rmse = calculate_rmse(clean_target, pred_clean_np)
            rmse_list.append(img_rmse)

            # Collect data for failure analysis
            # Flatten arrays to pixel level
            c_flat = clean_target.flatten()
            p_flat = pred_clean_np.flatten()
            n_flat = noisy.cpu().numpy().flatten()

            # Error magnitude
            diff = np.abs(c_flat - p_flat)

            flat_errors.append(diff)
            flat_noisy.append(n_flat)
            flat_clean.append(c_flat)

    # Calculate Final Metric
    final_metric = np.mean(rmse_list)
    # Print full precision as required
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    # Concatenate all pixel data from all validation images
    all_e = np.concatenate(flat_errors)
    all_n = np.concatenate(flat_noisy)
    all_c = np.concatenate(flat_clean)

    # Calculate correlations using numpy
    # Correlation between Error and Noisy Input Intensity
    corr_noisy = np.corrcoef(all_e, all_n)[0, 1]

    # Correlation between Error and Clean Target Intensity
    corr_clean = np.corrcoef(all_e, all_c)[0, 1]

    print("Failure Analysis Correlations:")
    print(f"Error vs Input Intensity: {corr_noisy}")
    print(f"Error vs Target Intensity: {corr_clean}")

    # 4. Submission
    # Threshold defined in task description
    THRESHOLD = 0.0076658159

    if final_metric < THRESHOLD:
        trainer.generate_submission()
    else:
        print(
            f"Validation metric {final_metric} is not lower than {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
