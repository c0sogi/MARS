import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import DEVICE, SUBMISSION_DIR, METADATA_DIR
from library.utils import set_seed, get_logger
from library.data_loader import get_data_loaders
from library.train_eval import train_fold, validate, predict

# Initialize Logger
logger = get_logger("runfile")


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and incidence angle.
    """
    model.eval()
    all_targets = []
    all_probs = []
    all_angles = []

    # Collect data
    with torch.no_grad():
        for inputs, angles, labels in val_loader:
            inputs = inputs.to(device)
            angles_gpu = angles.to(device)

            outputs = model(inputs, angles_gpu)
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_angles.append(angles.numpy())  # Keep on CPU

    # Concatenate
    y_true = np.concatenate(all_targets).flatten()
    y_pred = np.concatenate(all_probs).flatten()
    angles = np.concatenate(all_angles).flatten()

    # Calculate Error Magnitude (Absolute Error)
    errors = np.abs(y_true - y_pred)

    # Calculate Correlation with Incidence Angle
    # Note: Angles might have been imputed, but they are numerical now.
    if len(errors) > 1:
        corr, p_value = pearsonr(errors, angles)
    else:
        corr = 0.0

    print(f"Correlation between Error Magnitude and Incidence Angle: {corr:.6f}")
    return corr


def main():
    # 1. Setup
    set_seed(42)
    logger.info("Starting runfile execution...")

    # 2. Data Loading
    # Using the fixed split from metadata as required
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        load_cached_data=True, batch_size=32
    )

    # 3. Training
    # We train a single model on the 'train' split and validate on 'val' split.
    # Using 35 epochs to balance speed and convergence (early stopping usually hits earlier).
    logger.info("Training model...")
    # train_fold returns the model with best weights loaded
    model = train_fold(
        fold_idx=0,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=35,
        patience=10,
        device=DEVICE,
    )

    # 4. Validation Metric
    logger.info("Calculating final validation metric...")
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    logger.info("Performing failure analysis...")
    analyze_failures(model, val_loader, DEVICE)

    # 6. Submission Logic
    THRESHOLD = 0.15744295919935183

    if val_loss < THRESHOLD:
        logger.info(
            f"Validation metric {val_loss} < {THRESHOLD}. Generating submission..."
        )

        # Generate Predictions
        test_probs = predict(model, test_loader, DEVICE)

        # Load Test IDs
        # We read from metadata/test.csv to ensure alignment
        test_meta_path = os.path.join(METADATA_DIR, "test.csv")
        if os.path.exists(test_meta_path):
            df_test_meta = pd.read_csv(test_meta_path)
            test_ids = df_test_meta["id"].values
        else:
            logger.error(f"Test metadata file not found at {test_meta_path}")
            return

        # Verify lengths
        if len(test_ids) != len(test_probs):
            logger.error(
                f"Mismatch in test IDs ({len(test_ids)}) and predictions ({len(test_probs)})"
            )
            return

        # Create Submission DataFrame
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs})

        # Save
        df_sub.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")

    else:
        logger.info(
            f"Validation metric {val_loss} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
