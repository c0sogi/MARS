import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from library.config import Config
from library.utils import seed_everything, get_logger
from library.model import ACICNN
from library.data_loader import get_data, get_fold_loaders, get_test_loader

logger = get_logger("inference")


def predict_test():
    """
    Generates predictions for the test set using the ensemble of trained models.

    Steps:
    1. Loads cached data.
    2. Iterates through each of the 5 folds.
    3. For each fold:
       - Re-derives the specific scaler and imputation values used during training
         to ensure leak-free consistency.
       - Loads the trained model checkpoint.
       - Generates predictions on the test set (without TTA).
    4. Averages predictions across all folds.
    5. Saves the result to submission.csv.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)
    Config.setup()

    logger.info("Starting Inference Pipeline...")

    # Load Data (Cached)
    # We need the full training data to reconstruct the scalers for each fold correctly
    data = get_data(load_cached_data=True)
    ids_test = data["ids_test"]

    # Initialize accumulator for ensemble predictions
    # Shape: (N_test,)
    ensemble_preds = np.zeros(len(ids_test), dtype=np.float32)

    # Iterate through all folds
    for fold_idx in range(Config.NUM_FOLDS):
        logger.info(f"Processing Fold {fold_idx}/{Config.NUM_FOLDS - 1}...")

        # 1. Reconstruct Preprocessing Stats
        # We call get_fold_loaders to get the exact scaler and imputation value
        # used for this fold's training. We discard the train/val loaders.
        _, _, scaler, imp_val = get_fold_loaders(
            fold_idx, data, batch_size=Config.BATCH_SIZE
        )

        # 2. Create Test Loader for this Fold
        # This applies the fold-specific normalization to the test set
        test_loader = get_test_loader(
            data, scaler, imp_val, batch_size=Config.BATCH_SIZE
        )

        # 3. Load Model
        model = ACICNN().to(Config.DEVICE)
        checkpoint_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold_idx}.pth")

        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found for fold {fold_idx}: {checkpoint_path}")
            raise FileNotFoundError(f"Missing checkpoint for fold {fold_idx}")

        state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        # 4. Generate Predictions
        fold_preds = []

        with torch.no_grad():
            for imgs, raw_angs, norm_angs in test_loader:
                imgs = imgs.to(Config.DEVICE)
                raw_angs = raw_angs.to(Config.DEVICE)
                norm_angs = norm_angs.to(Config.DEVICE)

                # Forward pass
                # Output shape: (Batch_Size, 1) -> Squeeze to (Batch_Size,)
                outputs = model(imgs, raw_angs, norm_angs).squeeze(1)

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.append(probs)

        # Concatenate batches
        fold_preds_flat = np.concatenate(fold_preds)

        # Add to ensemble (simple average)
        ensemble_preds += fold_preds_flat / Config.NUM_FOLDS

    logger.info("Inference complete. Saving submission...")

    # Save Submission
    submission = pd.DataFrame({"id": ids_test, "is_iceberg": ensemble_preds})

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)

    logger.info(f"Submission saved to {sub_path}")
    logger.info("Preview of submission:")
    logger.info(submission.head().to_string())
