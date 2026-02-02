import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.train import Trainer
from library.predict import InferenceEngine
from library.data import prepare_data, DenoisingDataset


def run_pipeline():
    # --- 1. Configuration Setup ---
    # Modify configuration for this run
    Config.EPOCHS = 30  # Sufficient for convergence on this dataset size
    Config.BATCH_SIZE = 64

    # Ensure reproducibility
    Config.set_seed(Config.SEED)

    print(f"Running pipeline on device: {Config.DEVICE}")

    # --- 2. Model Training ---
    print("\n=== Starting Training Phase ===")
    trainer = Trainer()
    # Train the model. load_cached_data=True allows using pre-computed patches if available.
    trainer.train(load_cached_data=True)

    # --- 3. Validation and Metric Calculation ---
    print("\n=== Starting Validation Phase ===")

    # Load the best model weights saved during training
    model = trainer.model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )

    model.eval()

    # Prepare validation data
    val_data = prepare_data(
        Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH, load_cached_data=True
    )
    val_dataset = DenoisingDataset(val_data, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    total_squared_error = 0.0
    total_pixels = 0

    # Lists for failure analysis
    patch_rmses = []
    patch_means = []
    patch_stds = []

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(Config.DEVICE)
            clean = clean.to(Config.DEVICE)

            # Predict noise residual
            pred_noise = model(noisy)

            # Reconstruct clean image: Clean = Noisy - Noise
            pred_clean = noisy - pred_noise

            # Clamp values to valid range [0, 1]
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

            # Calculate squared error
            diff = pred_clean - clean
            squared_diff = diff**2

            # Accumulate global stats
            total_squared_error += squared_diff.sum().item()
            total_pixels += squared_diff.numel()

            # --- Failure Analysis Data ---
            # Compute RMSE per patch for correlation analysis
            # squared_diff shape: (B, 1, H, W)
            mse_per_patch = (
                squared_diff.view(noisy.size(0), -1).mean(dim=1).cpu().numpy()
            )
            rmse_per_patch = np.sqrt(mse_per_patch)

            # Compute Input stats per patch
            mean_per_patch = noisy.view(noisy.size(0), -1).mean(dim=1).cpu().numpy()
            std_per_patch = noisy.view(noisy.size(0), -1).std(dim=1).cpu().numpy()

            patch_rmses.extend(rmse_per_patch)
            patch_means.extend(mean_per_patch)
            patch_stds.extend(std_per_patch)

    # Compute Final RMSE
    final_rmse = np.sqrt(total_squared_error / total_pixels)
    print(f"Final Validation Metric: {final_rmse}")

    # --- 4. Failure Analysis ---
    print("\n=== Failure Analysis ===")
    if len(patch_rmses) > 1:
        # Calculate Pearson correlation
        corr_mean = np.corrcoef(patch_rmses, patch_means)[0, 1]
        corr_std = np.corrcoef(patch_rmses, patch_stds)[0, 1]

        print(f"Correlation (Error Magnitude vs Input Mean Intensity): {corr_mean}")
        print(f"Correlation (Error Magnitude vs Input Std Deviation): {corr_std}")
    else:
        print("Insufficient data for failure analysis.")

    # --- 5. Submission Generation ---
    print("\n=== Submission Check ===")
    threshold = 0.016654925420880318

    if final_rmse < threshold:
        print(
            f"Validation metric passed threshold ({final_rmse} < {threshold}). Generating submission..."
        )
        engine = InferenceEngine()
        engine.run()
    else:
        print(
            f"Validation metric failed threshold ({final_rmse} >= {threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
