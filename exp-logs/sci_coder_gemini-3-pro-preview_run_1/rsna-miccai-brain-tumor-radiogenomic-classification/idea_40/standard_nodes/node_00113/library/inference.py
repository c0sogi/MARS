import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    MODELS_DIR,
    PLANES,
    N_FOLDS,
    BATCH_SIZE,
    SUBMISSION_PATH,
    METADATA_DIR,
)
from library.utils import (
    get_logger,
    save_predictions,
    seed_everything,
)
from library.preprocessing import process_dataset
from library.dataset import RASSEDataset, get_transforms
from library.model import ExpertNet


def predict_loader(model, loader, device):
    """
    Runs inference on a DataLoader using a specific model.
    Returns raw probabilities (sigmoid applied to logits).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds.extend(probs)

    return np.array(preds)


def run_inference(batch_size=BATCH_SIZE, num_workers=2, load_cached_data=True):
    """
    Orchestrates the full inference pipeline:
    1. Loads and processes test data (with caching).
    2. Iterates through all Expert Planes (Lower, Center, Upper).
    3. Iterates through all Folds (0-4) for each plane.
    4. Aggregates predictions via averaging (Consensus).
    5. Saves the final submission file.
    """
    logger = get_logger("inference")
    seed_everything()

    logger.info("Starting Inference Pipeline...")

    # 1. Load Test Metadata
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Test metadata not found at {test_meta_path}")

    df_test = pd.read_csv(test_meta_path)
    logger.info(f"Loaded test metadata: {len(df_test)} subjects.")

    # 2. Process Data
    # process_dataset handles the caching logic internally (load if exists, else compute & save)
    # It returns images of shape (N, 3, H, W, 3) where dim 1 is [Lower, Center, Upper]
    images, ids, _ = process_dataset(
        df_test, load_cached_data=load_cached_data, save_name="test"
    )

    num_samples = len(ids)
    # Accumulator for consensus predictions
    final_probs = np.zeros(num_samples, dtype=np.float64)
    model_count = 0

    # 3. Iterate through Experts (Planes)
    # PLANES = {"lower": -0.15, "center": 0.0, "upper": 0.15}
    for plane_name in PLANES.keys():
        logger.info(f"Processing Expert Plane: {plane_name}")

        # Create Dataset and Loader for this specific plane
        # The dataset selects the correct slice from the 'images' array based on plane_name
        test_dataset = RASSEDataset(
            images,
            ids,
            labels=None,
            plane_name=plane_name,
            transform=get_transforms(phase="test"),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # 4. Iterate through Folds
        for fold_idx in range(N_FOLDS):
            model_filename = f"best_model_{plane_name}_fold{fold_idx}.pth"
            model_path = os.path.join(MODELS_DIR, model_filename)

            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}. Skipping.")
                continue

            logger.info(f"Loading model: {model_filename}")

            # Initialize Model
            model = ExpertNet().to(DEVICE)

            # Load Weights
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)

            # Generate Predictions
            probs = predict_loader(model, test_loader, DEVICE)

            # Accumulate
            final_probs += probs
            model_count += 1

            # Clean up to save memory
            del model
            del state_dict
            torch.cuda.empty_cache()

    # 5. Aggregate and Save
    if model_count == 0:
        logger.error("No models were loaded. Cannot generate predictions.")
        return

    logger.info(f"Aggregating predictions from {model_count} models.")
    avg_probs = final_probs / model_count

    logger.info(f"Saving submission to {SUBMISSION_PATH}...")
    save_predictions(ids, avg_probs, SUBMISSION_PATH)

    logger.info("Inference completed successfully.")
