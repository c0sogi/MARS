import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from provided library files
import library.config as config
from library.utils import seed_everything, get_logger
from library.data_loader import get_datasets
from library.model_arch import MNSHDNetwork
from library.trainer import run_training
from library.inference import generate_submission

# Initialize Logger
logger = get_logger("runfile")


def perform_validation_analysis(model, val_dataset):
    """
    Evaluates the model on the validation set, calculates the metric,
    and performs failure analysis.
    """
    logger.info("Starting Validation Analysis...")

    # Create DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    model.eval()
    all_probs = []
    all_targets = []

    # Inference Loop
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(config.DEVICE)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate results
    y_pred = np.concatenate(all_probs).flatten()
    y_true = np.concatenate(all_targets).flatten()

    # 1. Calculate Metric
    try:
        auc_score = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_score = 0.5

    print(f"Final Validation Metric: {auc_score}")

    # 2. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Extract features from the validation dataset tensors for correlation
    # X shape: (N, Channels, H, W)
    # We calculate global mean and std of the pixel intensities for each subject
    # to see if signal intensity correlates with error.
    X_val = val_dataset.X

    # Compute stats per sample (axis 1,2,3 are Channel, H, W)
    # Using float64 for precision during stats calculation
    mean_intensities = np.mean(X_val, axis=(1, 2, 3), dtype=np.float64)
    std_intensities = np.std(X_val, axis=(1, 2, 3), dtype=np.float64)

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "mean_intensity": mean_intensities,
            "std_intensity": std_intensities,
            "true_label": y_true,
        }
    )

    # Calculate correlation matrix
    correlation = analysis_df.corr()["error"].drop("error")

    print("\nFailure Analysis - Correlation between Error and Input Features:")
    print(correlation)
    print("-" * 30)

    return auc_score


def main():
    # 1. Setup
    seed_everything(config.SEED)

    # 2. Training
    # run_training handles the full training loop and saves 'best_model.pth'
    logger.info("Initiating Training Pipeline...")
    run_training()

    # 3. Load Best Model for Analysis
    logger.info(f"Loading best model from {config.MODEL_SAVE_PATH}...")
    model = MNSHDNetwork().to(config.DEVICE)

    if os.path.exists(config.MODEL_SAVE_PATH):
        state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        model.load_state_dict(state_dict)
    else:
        logger.error("Model file not found. Training may have failed.")
        return

    # 4. Load Validation Data
    # get_datasets returns (train, val, test)
    _, val_dataset, _ = get_datasets(load_cached_data=True)

    # 5. Validation & Failure Analysis
    val_auc = perform_validation_analysis(model, val_dataset)

    # 6. Submission
    # Threshold defined in task requirements
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(
            model_path=config.MODEL_SAVE_PATH,
            output_path=config.SUBMISSION_FILE_PATH,
            load_cached_data=True,
        )
    else:
        logger.warning(
            f"Validation AUC ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
