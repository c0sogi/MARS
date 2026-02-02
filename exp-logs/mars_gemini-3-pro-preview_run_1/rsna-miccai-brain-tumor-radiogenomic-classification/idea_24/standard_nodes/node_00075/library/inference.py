import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_device, setup_logger
from library.data_processing import get_centroids_with_caching
from library.dataset import BraTSDataset, get_transforms
from library.model import CAWIVModel
from library.trainer import predict


def predict_test_set(load_cached_data=True):
    """
    Runs inference on the test set using trained models from all folds.
    Generates the submission.csv file.

    Args:
        load_cached_data (bool): Whether to load centroids from cache.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "inference.log"))

    logger.info("Starting Inference Pipeline...")

    # 2. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        logger.error(f"Test metadata not found at {Config.TEST_METADATA}")
        return

    df_test = pd.read_csv(Config.TEST_METADATA)
    logger.info(f"Loaded test metadata with {len(df_test)} subjects.")

    # 3. Prepare Centroids (Cache)
    # Ensure consistent preprocessing with training
    logger.info("Processing/Loading Test Centroids...")
    centroids_test = get_centroids_with_caching(
        df_test,
        Config.INPUT_DIR,
        cache_name="centroids_test",
        load_cached_data=load_cached_data,
    )

    # 4. Setup Dataset and Loader
    # Use validation transforms (no augmentation, just ToTensor)
    test_transform = get_transforms(mode="val")

    test_ds = BraTSDataset(
        df_test, centroids_test, Config.INPUT_DIR, transform=test_transform, mode="test"
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Crucial for order
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Inference Loop over Folds
    # We will accumulate probabilities in a dictionary to ensure ID alignment
    # Initialize with 0.0
    accumulated_probs = {sid: 0.0 for sid in df_test["BraTS21ID"].values}
    models_found = 0

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.CACHE_DIR, f"best_model_fold{fold}.pth")

        if not os.path.exists(model_path):
            logger.warning(
                f"Model for Fold {fold} not found at {model_path}. Skipping."
            )
            continue

        logger.info(f"Predicting with model from Fold {fold}...")

        # Load Model
        model = CAWIVModel(model_name=Config.BACKBONE, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Generate Predictions
        # predict returns (ids, probs)
        ids, probs = predict(model, test_loader, device)

        # Accumulate
        for sid, prob in zip(ids, probs):
            accumulated_probs[sid] += prob

        models_found += 1

    if models_found == 0:
        logger.error("No trained models found. Cannot generate submission.")
        return

    # 6. Average and Format
    logger.info(f"Averaging predictions from {models_found} models...")

    final_results = []
    for sid, total_prob in accumulated_probs.items():
        avg_prob = total_prob / models_found
        final_results.append({"BraTS21ID": sid, "MGMT_value": avg_prob})

    submission_df = pd.DataFrame(final_results)

    # Ensure column order and sorting
    submission_df = submission_df[["BraTS21ID", "MGMT_value"]]
    submission_df.sort_values("BraTS21ID", inplace=True)

    # 7. Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Preview:\n{submission_df.head()}")
