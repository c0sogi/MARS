import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library import utils, model, data_loader


def generate_predictions(load_cached_data=True):
    """
    Generates predictions for the test set using the trained models from all folds.
    Averages the probabilities across folds (Ensembling).
    Saves the result to submission.csv.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data for the test set.
    """
    # Setup
    utils.set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    log_file = os.path.join(Config.WORKING_DIR, "inference.log")
    logger = utils.setup_logger(log_file, name="inference")

    logger.info("Starting inference pipeline...")
    logger.info(f"Device: {device}")

    # Load Test Data
    # Note: Test loader handles imputation of missing angles internally using training stats
    test_loader = data_loader.get_test_loader(load_cached_data=load_cached_data)

    # Storage for predictions from each fold
    # List of numpy arrays, each shape (N_test, 1)
    fold_probabilities = []

    # Storage for IDs (order is preserved in DataLoader)
    test_ids = []
    ids_collected = False

    # Iterate over all folds defined in configuration
    for fold in range(Config.N_FOLDS):
        checkpoint_path = os.path.join(
            Config.WORKING_DIR, f"model_best_fold_{fold}.pth"
        )

        if not os.path.exists(checkpoint_path):
            logger.info(
                f"Checkpoint for Fold {fold} not found at {checkpoint_path}. Skipping."
            )
            continue

        logger.info(f"Processing Fold {fold}...")

        # Initialize Model
        net = model.IAMSI_CNN().to(device)

        # Load Weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

        # Set to Evaluation Mode
        # IMPORTANT: This activates the averaging of the multi-sample dropout branches
        # within the model's forward() method.
        net.eval()

        current_fold_preds = []

        with torch.no_grad():
            for i, (images, angles, ids) in enumerate(test_loader):
                images = images.to(device)
                angles = angles.to(device)

                # Forward pass
                # In eval mode, model returns (Batch, 1) - averaged logits from dropout branches
                logits = net(images, angles)

                # Apply Sigmoid to get probabilities (Logits -> Probability)
                probs = torch.sigmoid(logits)

                current_fold_preds.append(probs.cpu().numpy())

                # Collect IDs only during the first successful fold iteration
                # The DataLoader order is deterministic (shuffle=False)
                if not ids_collected:
                    test_ids.extend(ids)

        # Mark IDs as collected so we don't duplicate them
        if not ids_collected and len(test_ids) > 0:
            ids_collected = True

        # Concatenate batches for this fold
        # Result shape: (N_test, 1)
        fold_probs_concat = np.concatenate(current_fold_preds, axis=0)
        fold_probabilities.append(fold_probs_concat)

    if not fold_probabilities:
        logger.info("No predictions were generated. Ensure checkpoints exist.")
        return

    # Convert to numpy array: (N_Folds, N_Test, 1)
    fold_probabilities = np.array(fold_probabilities)

    # Average across folds (Soft Voting / Ensembling)
    # Shape: (N_Test, 1)
    avg_probs = np.mean(fold_probabilities, axis=0)

    # Flatten to 1D array
    avg_probs = avg_probs.flatten()

    # Validation check
    if len(test_ids) != len(avg_probs):
        logger.info(
            f"Error: Mismatch between ID count ({len(test_ids)}) and prediction count ({len(avg_probs)})"
        )
        return

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_probs})

    # Ensure output directory exists
    Config.setup_directories()

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Inference complete.")
    logger.info(f"Submission saved to: {Config.SUBMISSION_PATH}")
    logger.info(f"Total predictions: {len(submission_df)}")
    logger.info(f"Sample:\n{submission_df.head()}")
