import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_logger
from library.data import get_test_loader
from library.model import HybridEfficientNet


def predict_with_tta(model, loader, device):
    """
    Performs inference with Test Time Augmentation (TTA).
    TTA Strategy: Original, Horizontal Flip, Vertical Flip, Rotate 180.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        dict: Dictionary mapping image_names to predicted probabilities.
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            image_names = batch["image_name"]

            # TTA 1: Original
            logits_1 = model(images, tabular)
            probs_1 = torch.sigmoid(logits_1)

            # TTA 2: Horizontal Flip
            images_hf = torch.flip(images, dims=[3])
            logits_2 = model(images_hf, tabular)
            probs_2 = torch.sigmoid(logits_2)

            # TTA 3: Vertical Flip
            images_vf = torch.flip(images, dims=[2])
            logits_3 = model(images_vf, tabular)
            probs_3 = torch.sigmoid(logits_3)

            # TTA 4: Rotate 180 (Horizontal + Vertical Flip)
            images_rot = torch.flip(images, dims=[2, 3])
            logits_4 = model(images_rot, tabular)
            probs_4 = torch.sigmoid(logits_4)

            # Average probabilities
            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            # Store results
            avg_probs_np = avg_probs.cpu().numpy().flatten()
            for name, prob in zip(image_names, avg_probs_np):
                predictions[name] = prob

    return predictions


def generate_submission(load_cached_data=True):
    """
    Orchestrates the ensemble inference process.
    Loads models for all folds, performs TTA inference, averages results,
    and saves the submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    logger = get_logger("Inference")
    device = torch.device(Config.DEVICE)

    logger.info("Loading test data...")
    test_loader, num_tabular_features = get_test_loader(
        load_cached_data=load_cached_data
    )

    # Dictionary to aggregate predictions across folds
    # Key: image_name, Value: accumulated probability
    ensemble_preds = {}

    # Iterate over all folds
    for fold in range(Config.NUM_FOLDS):
        logger.info(f"Processing Fold {fold}...")

        # Initialize model
        model = HybridEfficientNet(
            model_name=Config.MODEL_NAME,
            pretrained=False,  # No need to download weights, we load checkpoint
            num_classes=Config.NUM_CLASSES,
            num_tabular_features=num_tabular_features,
            tabular_hidden_dim=Config.TABULAR_HIDDEN_DIM,
            final_dropout=Config.FINAL_DROPOUT,
        )
        model.to(device)

        # Load checkpoint
        checkpoint_path = os.path.join(Config.WORKING_DIR, f"fold_{fold}_best.pth")
        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Checkpoint for fold {fold} not found at {checkpoint_path}. Skipping."
            )
            continue

        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        # Run Inference
        fold_preds = predict_with_tta(model, test_loader, device)

        # Accumulate
        for img_name, prob in fold_preds.items():
            if img_name not in ensemble_preds:
                ensemble_preds[img_name] = 0.0
            ensemble_preds[img_name] += prob

    # Average predictions and prepare dataframe
    final_data = []
    num_models = Config.NUM_FOLDS

    # Note: If a fold was skipped, this division assumes 5 folds.
    # In a strict pipeline, all folds should exist.

    for img_name, total_prob in ensemble_preds.items():
        avg_prob = total_prob / num_models
        final_data.append({"image_name": img_name, "target": avg_prob})

    df_submission = pd.DataFrame(final_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save submission
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Total predictions generated: {len(df_submission)}")
