import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import TumorDataset, get_transforms
from library.network import DenseNet121GeM
from library.utils import setup_logger


def run_inference(device=None):
    """
    Executes the inference pipeline:
    1. Loads test data.
    2. Loads the ensemble of trained models.
    3. Performs inference with Test-Time Augmentation (TTA).
    4. Saves the submission file.

    Args:
        device (torch.device, optional): The device to run inference on.
                                         Defaults to Config.DEVICE.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    # Setup logger
    logger = setup_logger(
        "Inference", os.path.join(Config.WORKING_DIR, "inference.log")
    )
    logger.info("Starting Inference Pipeline...")

    # ==========================================
    # 1. Load Test Data
    # ==========================================
    logger.info(f"Loading test metadata from {Config.TEST_METADATA_PATH}")
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debugging: Subsample if configured
    if Config.DEBUG_SAMPLE_SIZE is not None:
        logger.info(
            f"Debug mode: Subsampling test set to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Create Dataset and DataLoader
    # We use the 'test' transform which only does CenterCrop, Norm, and ToTensor
    test_dataset = TumorDataset(test_df, transform=get_transforms("test"), phase="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # ==========================================
    # 2. Load Ensemble Models
    # ==========================================
    models = []
    logger.info(f"Loading models for {Config.NUM_FOLDS} folds...")

    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(
            Config.WORKING_DIR, f"{Config.MODEL_NAME}_fold{fold}_best.pth"
        )

        if not os.path.exists(model_path):
            logger.warning(
                f"Checkpoint for fold {fold} not found at {model_path}. Skipping this fold."
            )
            continue

        # Initialize model architecture
        # pretrained=False because we are loading custom weights
        model = DenseNet121GeM(pretrained=False)

        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.error(f"Failed to load weights for fold {fold}: {e}")
            continue

        model.to(device)
        model.eval()
        models.append(model)
        logger.info(f"Successfully loaded model for fold {fold}.")

    if not models:
        raise RuntimeError(
            "No trained models were loaded. Cannot proceed with inference."
        )

    logger.info(f"Ensemble loaded with {len(models)} models.")

    # ==========================================
    # 3. Inference Loop with TTA
    # ==========================================
    logger.info(f"Starting inference on {len(test_dataset)} images...")
    if Config.TTA_ENABLED:
        logger.info("Test-Time Augmentation (TTA) is ENABLED (4 views).")
    else:
        logger.info("Test-Time Augmentation (TTA) is DISABLED.")

    all_preds = []
    all_ids = []

    # Define TTA transformations (applied on tensors)
    # Images are (B, C, H, W). H=dim 2, W=dim 3.
    tta_transforms = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, [3]),  # Horizontal Flip
        lambda x: torch.flip(x, [2]),  # Vertical Flip
        lambda x: torch.rot90(x, 1, [2, 3]),  # Rotate 90 degrees
    ]

    # If TTA is disabled, only use the identity transform
    if not Config.TTA_ENABLED:
        tta_transforms = [tta_transforms[0]]

    total_views = len(tta_transforms)
    total_models = len(models)
    normalization_factor = total_views * total_models

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            images = data["image"].to(device, dtype=torch.float)
            ids = data["id"]

            # Accumulator for probabilities: Shape (Batch_Size,)
            batch_probs = torch.zeros(images.size(0), device=device)

            # Iterate over each TTA view
            for transform_func in tta_transforms:
                # Apply geometric transformation
                augmented_images = transform_func(images)

                # Pass through each model in the ensemble
                for model in models:
                    logits = model(augmented_images).view(-1)
                    probs = torch.sigmoid(logits)
                    batch_probs += probs

            # Average the probabilities
            avg_probs = batch_probs / normalization_factor

            # Store results
            all_preds.extend(avg_probs.cpu().numpy())
            all_ids.extend(ids)

            # Logging progress periodically
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    # ==========================================
    # 4. Save Submission
    # ==========================================
    submission_df = pd.DataFrame({"id": all_ids, "label": all_preds})

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")

    return submission_df
