import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, calculate_weighted_loss
from library.data import get_dataloaders
from library.model import CalibratedSequenceModel
from library.engine import fit, validate, inference


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to find correlations
    between error magnitude and input features (e.g., slice count).
    """
    logger = get_logger(name="failure_analysis")
    logger.info("Starting failure analysis...")

    model.eval()
    results = []

    # Weights for the loss function (matching library/utils.py)
    weights = np.array([7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    with torch.no_grad():
        for images, targets, uids in val_loader:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(images)
            outputs = torch.clamp(outputs, 1e-7, 1.0 - 1e-7)

            # Convert to numpy
            preds_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Calculate per-sample loss
            # Loss = -w * [y log p + (1-y) log (1-p)]
            # We sum across the 8 classes for the patient
            term1 = targets_np * np.log(preds_np)
            term2 = (1 - targets_np) * np.log(1 - preds_np)
            loss_matrix = -weights * (term1 + term2)
            sample_losses = np.sum(loss_matrix, axis=1)  # Sum weighted loss per patient

            # Get metadata features (Slice Count)
            # Access the dataset's path_map via the loader
            path_map = val_loader.dataset.path_map

            for i, uid in enumerate(uids):
                # Slice count is the number of files in the directory
                slice_count = len(path_map.get(uid, []))

                results.append(
                    {
                        "StudyInstanceUID": uid,
                        "loss": sample_losses[i],
                        "slice_count": slice_count,
                    }
                )

    # Create DataFrame
    df = pd.DataFrame(results)

    if len(df) > 1:
        # Calculate correlation
        correlation = df["loss"].corr(df["slice_count"])
        logger.info(
            f"Correlation between Error (Loss) and Slice Count: {correlation:.4f}"
        )

        # Additional stats
        logger.info(f"Mean Loss: {df['loss'].mean():.4f}")
        logger.info(f"Max Loss: {df['loss'].max():.4f}")
    else:
        logger.warning("Not enough samples for correlation analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger(name="runfile")
    device = Config.DEVICE
    logger.info(f"Running on device: {device}")

    # 2. Data Loading
    # Using cached data for speed as requested
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    logger.info("Initializing model...")
    model = CalibratedSequenceModel()
    model.to(device)

    # 4. Training
    # fit() handles the training loop, checkpointing, and returns the best model
    logger.info("Starting training...")
    model = fit(model, train_loader, val_loader, device, epochs=Config.EPOCHS)

    # 5. Final Validation
    # We validate again on the best model to get the exact metric for the log
    logger.info("Performing final validation...")
    _, val_metric = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 7. Conditional Submission
    # Threshold defined in task description
    THRESHOLD = 0.15364714496434773

    if val_metric < THRESHOLD:
        logger.info(
            f"Validation metric ({val_metric:.6f}) is better than threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        inference(model, test_loader, device)
    else:
        logger.info(
            f"Validation metric ({val_metric:.6f}) did not beat threshold ({THRESHOLD:.6f}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
