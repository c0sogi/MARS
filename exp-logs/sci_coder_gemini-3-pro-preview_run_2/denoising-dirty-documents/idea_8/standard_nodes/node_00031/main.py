import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import seed_everything, get_logger, load_checkpoint
from library.dataset import DenoisingDataset
from library.model import CoConvNeXtUNet, train_model
from library.inference import generate_submission


def main():
    # --- 1. Configuration for Fast Baseline ---
    # Adjust epochs and sampling density to ensure completion within the time limit
    # while maintaining enough capacity to learn useful features.
    Config.NUM_EPOCHS = 40
    Config.PATCHES_PER_IMAGE = 50

    # Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("RunFile")

    logger.info(f"Starting runfile.py on {device}")
    logger.info(
        f"Config: Epochs={Config.NUM_EPOCHS}, Patches/Img={Config.PATCHES_PER_IMAGE}"
    )

    # --- 2. Data Loading ---
    # Load cached data = True to save time on IO and preprocessing
    train_dataset = DenoisingDataset(
        Config.TRAIN_CSV, mode="train", load_cached_data=True
    )
    val_dataset = DenoisingDataset(Config.VAL_CSV, mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Validation images vary in size, so batch_size=1 is safer
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Model Initialization ---
    model = CoConvNeXtUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_filters=Config.BASE_FILTERS,
    ).to(device)

    # --- 4. Training ---
    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, device)

    # --- 5. Validation & Failure Analysis ---
    logger.info("Loading best model for validation and failure analysis...")
    # Load best weights to ensure we evaluate the optimal state
    load_checkpoint(Config.MODEL_PATH, model, device=device)
    model.eval()

    total_mse = 0.0
    total_pixels = 0

    # Lists for failure analysis
    errors_rmse = []
    input_means = []
    input_stds = []

    with torch.no_grad():
        for noisy, clean, _ in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Predict noise residual
            noise_pred = model(noisy)

            # Reconstruct clean image
            clean_pred = noisy - noise_pred
            clean_pred = torch.clamp(clean_pred, 0, 1)

            # Calculate metrics
            mse_sum = F.mse_loss(clean_pred, clean, reduction="sum").item()
            num_pix = clean.numel()

            total_mse += mse_sum
            total_pixels += num_pix

            # Per image stats for failure analysis
            img_rmse = np.sqrt(mse_sum / num_pix)
            errors_rmse.append(img_rmse)

            # Input features (calculate on CPU numpy)
            noisy_np = noisy.cpu().numpy()
            input_means.append(np.mean(noisy_np))
            input_stds.append(np.std(noisy_np))

    final_metric = np.sqrt(total_mse / total_pixels)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    if len(errors_rmse) > 1:
        # Pearson correlation coefficient matrix
        corr_mean = np.corrcoef(errors_rmse, input_means)[0, 1]
        corr_std = np.corrcoef(errors_rmse, input_stds)[0, 1]

        print(f"Failure Analysis - Correlation (Error vs Input Mean): {corr_mean:.4f}")
        print(f"Failure Analysis - Correlation (Error vs Input Std): {corr_std:.4f}")
    else:
        print("Insufficient validation samples for failure analysis.")

    # --- 6. Conditional Submission ---
    THRESHOLD = 0.0076658159

    if final_metric < THRESHOLD:
        logger.info(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        # Load Test Data
        test_dataset = DenoisingDataset(
            Config.TEST_CSV, mode="test", load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        generate_submission(test_loader, device)
    else:
        logger.info(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
