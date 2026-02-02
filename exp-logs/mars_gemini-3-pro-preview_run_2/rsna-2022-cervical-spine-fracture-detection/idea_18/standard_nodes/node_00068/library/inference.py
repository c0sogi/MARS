import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import RSNADataset, get_transforms, cache_image_paths
from library.model import CervicalSpineModel
from library.utils import seed_everything, get_logger, load_checkpoint


def predict(debug=False, load_cached_data=True):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        debug (bool): If True, runs on a small subset of the test data.
        load_cached_data (bool): If True, attempts to load image paths from cache.
    """
    # 1. Setup
    seed_everything(Config.seed)
    logger = get_logger("inference.log")
    logger.info("Starting inference...")

    # 2. Load Metadata
    test_df = pd.read_csv(Config.test_metadata_path)

    if debug:
        logger.info("DEBUG MODE: Truncating test dataset.")
        test_df = test_df.head(10)

    logger.info(f"Test set size: {len(test_df)}")

    # 3. Cache Image Paths
    # We use the existing caching logic for the test set
    test_paths_map = cache_image_paths(
        test_df, "test", load_cached_data=load_cached_data
    )

    # 4. Dataset & DataLoader
    test_dataset = RSNADataset(
        test_df,
        test_paths_map,
        phase="test",
        transform=get_transforms("test"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,  # Crucial: must be False to match UIDs
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 5. Model Initialization & Loading
    logger.info(f"Initializing model: {Config.backbone}")
    model = CervicalSpineModel()
    model.to(Config.device)

    checkpoint_path = os.path.join(Config.working_dir, "best_model.pth")
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path, device=Config.device)
    else:
        logger.warning(
            f"Checkpoint not found at {checkpoint_path}. Using random weights (expect poor performance)."
        )

    model.eval()

    # 6. Inference Loop
    all_preds = []

    # Target columns corresponding to model output indices 0-7
    # Order defined in Config.target_cols: ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    target_cols = Config.target_cols

    logger.info("Running prediction loop...")

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(Config.device, non_blocking=True)

            # Forward pass
            logits = model(images)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and numpy
            probs = probs.cpu().numpy()

            all_preds.append(probs)

    # Concatenate all batches
    if len(all_preds) > 0:
        predictions = np.concatenate(all_preds, axis=0)
    else:
        predictions = np.zeros((0, len(target_cols)))

    # 7. Format Submission
    logger.info("Formatting submission...")

    submission_rows = []
    study_uids = test_df["StudyInstanceUID"].values

    # Ensure we have the same number of predictions as studies
    if len(predictions) != len(study_uids):
        logger.error(
            f"Mismatch: {len(predictions)} predictions vs {len(study_uids)} studies."
        )

    for idx, uid in enumerate(study_uids):
        # Get the 8 probabilities for this study
        study_probs = predictions[idx]

        for class_idx, class_name in enumerate(target_cols):
            # Construct row_id: e.g., "1.2.3.4_C1"
            row_id = f"{uid}_{class_name}"
            prob = study_probs[class_idx]

            submission_rows.append({"row_id": row_id, "fractured": float(prob)})

    submission_df = pd.DataFrame(submission_rows)

    # 8. Save Submission
    output_dir = "./submission"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "submission.csv")

    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Submission shape: {submission_df.shape}")
    logger.info(f"First few rows:\n{submission_df.head()}")

    return submission_df
