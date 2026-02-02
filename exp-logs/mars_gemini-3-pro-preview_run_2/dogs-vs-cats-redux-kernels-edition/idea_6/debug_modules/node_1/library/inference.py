import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import DogCatDataset, get_valid_transforms
from library.models import build_model
from library.utils import load_checkpoint, get_logger

logger = get_logger("inference")


def predict_with_tta(model, test_loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    TTA involves averaging the predictions of the original image and its horizontally flipped version.
    Applies Sigmoid activation to convert logits to probabilities.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        dict: A dictionary mapping image ID to predicted probability (0-1).
    """
    model.eval()
    preds_dict = {}

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # 1. Forward pass on original images
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Forward pass on horizontally flipped images
            # Images are (B, C, H, W). Flip on W (dim 3).
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)

            # 3. Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            # Convert to numpy
            avg_probs_np = avg_probs.cpu().numpy().flatten()
            ids_np = ids.numpy().flatten()

            # Store results
            for img_id, prob in zip(ids_np, avg_probs_np):
                preds_dict[img_id] = prob

    return preds_dict


def run_inference():
    """
    Orchestrates the inference process:
    1. Loads test data.
    2. Iterates through all ensemble models (Architectures x Folds).
    3. Aggregates predictions.
    4. Saves submission file.
    """
    device = Config.DEVICE

    # Load Test Metadata
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_CSV}")

    test_df = pd.read_csv(Config.TEST_CSV)

    # Prepare Dataset and Loader
    # We use valid_transforms which resizes and normalizes
    test_dataset = DogCatDataset(
        test_df, transforms=get_valid_transforms(Config.IMG_SIZE), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
    )

    # Initialize dictionary to store accumulated probabilities
    # Using a dict ensures we map correctly by ID regardless of loader order
    aggregated_preds = {row["id"]: 0.0 for _, row in test_df.iterrows()}

    total_models = 0

    # Iterate over all architectures and folds
    for arch in Config.MODEL_ARCHS:
        for fold in range(Config.N_FOLDS):
            checkpoint_path = Config.get_checkpoint_path(arch, fold)

            if not os.path.exists(checkpoint_path):
                logger.warning(
                    f"Checkpoint not found: {checkpoint_path}. Skipping this model."
                )
                continue

            logger.info(f"Running inference with {arch} (Fold {fold})...")

            # Build model structure
            model = build_model(arch, pretrained=False, num_classes=1)

            # Load weights
            load_checkpoint(checkpoint_path, model, device=device)
            model.to(device)

            # Get predictions with TTA
            fold_preds = predict_with_tta(model, test_loader, device)

            # Accumulate predictions
            for img_id, prob in fold_preds.items():
                aggregated_preds[img_id] += prob

            total_models += 1

    if total_models == 0:
        logger.error("No models were found or loaded. Aborting inference.")
        return

    logger.info(f"Aggregating predictions from {total_models} models...")

    # Average the accumulated probabilities
    submission_data = []
    for img_id, total_prob in aggregated_preds.items():
        avg_prob = total_prob / total_models
        submission_data.append({"id": int(img_id), "label": avg_prob})

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Sort by ID to ensure consistent order
    submission_df = submission_df.sort_values("id")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
