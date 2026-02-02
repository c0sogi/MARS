import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import setup_logger, set_seed
from library.model import DPSCACNN
from library.data_loader import process_data, IcebergDataset


def predict_test(load_cached_data=True):
    """
    Performs inference on the test set using the trained models from all folds.
    Generates a submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # Setup logging
    log_file = os.path.join(Config.WORKING_DIR, "inference.log")
    logger = setup_logger("inference", log_file)
    logger.info("Starting inference pipeline...")

    # Set seed
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Test Data
    # We use process_data to ensure we get the IDs and consistent preprocessing
    logger.info("Loading test data...")
    _, _, _, X_test, ang_test, ids_test = process_data(
        load_cached_data=load_cached_data
    )

    # Create Dataset and Loader
    test_dataset = IcebergDataset(X_test, ang_test, labels=None, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Initialize array for ensemble predictions
    num_samples = len(ids_test)
    ensemble_probs = np.zeros(num_samples, dtype=np.float32)
    models_loaded = 0

    # 2. Iterate over folds for Ensemble
    for fold in range(Config.NUM_FOLDS):
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Checkpoint for Fold {fold} not found at {checkpoint_path}. Skipping."
            )
            continue

        logger.info(f"Loading model for Fold {fold}...")

        # Initialize model
        model = DPSCACNN().to(device)

        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        # Evaluation mode
        model.eval()

        fold_probs = []

        # Inference loop
        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Forward pass
                logits = model(images, angles)

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                # Flatten and store
                fold_probs.extend(probs.cpu().numpy().flatten())

        # Accumulate
        ensemble_probs += np.array(fold_probs)
        models_loaded += 1
        logger.info(f"Fold {fold} inference completed.")

    # 3. Average Predictions
    if models_loaded == 0:
        logger.error("No models were loaded. Cannot generate submission.")
        return

    ensemble_probs /= models_loaded
    logger.info(f"Averaged predictions from {models_loaded} models.")

    # 4. Generate Submission File
    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": ensemble_probs})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved successfully to {submission_path}")
