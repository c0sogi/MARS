import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config, seed_everything
from library.utils import get_logger
from library.train import run_fold
from library.dataset import get_dataset
from library.model import IcebergResNet


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("Runfile")

    # Speed up for baseline execution
    Config.NUM_EPOCHS = 15

    logger.info("Starting baseline execution...")

    # 2. Train Model
    # Using the provided library function which uses the fixed metadata split (Fold 0)
    model = run_fold(fold_idx=0)

    # 3. Validation & Failure Analysis
    logger.info("Performing validation and failure analysis...")
    device = torch.device(Config.DEVICE)
    model.eval()

    val_dataset = get_dataset("val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for images, angles, labels, _ in val_loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()

    # Calculate Metric
    final_metric = log_loss(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    # Calculate error magnitude
    errors = np.abs(all_targets - all_preds)

    # Extract features for correlation from raw dataset arrays
    # val_dataset.images is (N, 75, 75, 3) containing raw dB values
    raw_images = val_dataset.images
    raw_angles = val_dataset.angles

    b1_means = np.mean(raw_images[:, :, :, 0], axis=(1, 2))
    b2_means = np.mean(raw_images[:, :, :, 1], axis=(1, 2))

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": raw_angles,
            "b1_mean": b1_means,
            "b2_mean": b2_means,
        }
    )

    # Handle NaNs in angle if any (though dataset fills them)
    df_analysis.fillna(0, inplace=True)

    logger.info("Failure Analysis - Correlation with Error Magnitude:")
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    # 4. Submission
    THRESHOLD = 0.21099163245555455
    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric {final_metric:.6f} < {THRESHOLD}. Generating submission..."
        )

        test_dataset = get_dataset("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_ids = []
        test_preds = []

        with torch.no_grad():
            for images, angles, ids in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                # TTA: Original
                logits_orig = model(images, angles)
                probs_orig = torch.sigmoid(logits_orig)

                # TTA: Horizontal Flip
                images_h = torch.flip(images, dims=[3])
                logits_h = model(images_h, angles)
                probs_h = torch.sigmoid(logits_h)

                # TTA: Vertical Flip
                images_v = torch.flip(images, dims=[2])
                logits_v = model(images_v, angles)
                probs_v = torch.sigmoid(logits_v)

                # Average Probabilities
                probs_avg = (probs_orig + probs_h + probs_v) / 3.0

                test_preds.extend(probs_avg.cpu().numpy().flatten())
                test_ids.extend(ids)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation metric {final_metric:.6f} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
