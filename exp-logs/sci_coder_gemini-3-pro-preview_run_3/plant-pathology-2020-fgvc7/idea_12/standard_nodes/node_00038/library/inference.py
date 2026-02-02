import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.data import AppleDataset, get_transforms
from library.model import AppleDiseaseFPN
from library.utils import get_logger

logger = get_logger(__name__)


def predict_with_tta(model, loader, device):
    """
    Performs inference with Domain-Aware Test-Time Augmentation.
    Specifically applies Horizontal Flip TTA as defined in Config.

    Args:
        model: The PyTorch model in eval mode.
        loader: DataLoader for the test set.
        device: Computation device.

    Returns:
        tuple: (probabilities numpy array, list of image_ids)
    """
    model.eval()
    all_probs = []
    all_ids = []

    # Disable gradient calculation for inference efficiency
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # 1. Forward pass on original images
            logits_orig = model(images)
            probs_orig = torch.softmax(logits_orig, dim=1)

            # 2. Forward pass on horizontally flipped images
            # Tensor shape is (B, C, H, W). We flip along the width dimension (dim=3).
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = torch.softmax(logits_flip, dim=1)

            # 3. Average the probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0

            all_probs.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate results from all batches
    return np.concatenate(all_probs, axis=0), all_ids


def run_inference():
    """
    Main inference routine.
    - Loads test metadata.
    - Iterates through all defined architectures and folds.
    - Performs TTA inference.
    - Ensembles predictions via unweighted averaging.
    - Saves the final submission CSV.
    """
    # 1. Load Test Metadata
    if not os.path.exists(Config.test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {Config.test_csv_path}")

    test_df = pd.read_csv(Config.test_csv_path)
    logger.info(f"Loaded test metadata: {len(test_df)} images.")

    # 2. Initialize Ensemble Accumulator
    # We use a dictionary to safely map predictions to image_ids regardless of shuffle/order
    # Key: image_id, Value: np.array of shape (num_classes,)
    ensemble_preds = {
        img_id: np.zeros(Config.num_classes) for img_id in test_df["image_id"].values
    }

    total_models_used = 0

    # 3. Iterate over Model Architectures defined in Config
    for model_cfg in Config.models:
        model_name = model_cfg["name"]
        img_size = model_cfg["img_size"]
        batch_size = model_cfg["batch_size"]

        logger.info(f"Processing architecture: {model_name} (Input Size: {img_size})")

        # Create DataLoader for this specific image size
        # We use 'test' mode for transforms (Resize + Normalize, no augmentation)
        transform = get_transforms(data="test", img_size=img_size)
        dataset = AppleDataset(
            test_df, transform=transform, data_root=Config.input_root, is_test=True
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Iterate over Folds (0 to 4)
        # Assuming 5-fold CV based on the strategy description
        for fold in range(5):
            # Construct weights filename: e.g., tf_efficientnetv2_m_fold_0.pth
            weights_filename = f"{model_name}_fold_{fold}.pth"
            weights_path = os.path.join(Config.working_dir, weights_filename)

            if not os.path.exists(weights_path):
                logger.warning(
                    f"Weights file not found: {weights_path}. Skipping this fold."
                )
                continue

            logger.info(f"  -> Loading weights: {weights_filename}")

            # Initialize Model
            # pretrained=False because we are loading our own trained weights
            model = AppleDiseaseFPN(
                model_name=model_name,
                num_classes=Config.num_classes,
                pretrained=False,
            )

            # Load State Dict
            try:
                state_dict = torch.load(weights_path, map_location=Config.device)
                model.load_state_dict(state_dict)
            except Exception as e:
                logger.error(f"Failed to load weights {weights_path}: {e}")
                continue

            model.to(Config.device)

            # Predict with TTA
            probs, ids = predict_with_tta(model, loader, Config.device)

            # Accumulate Predictions
            for img_id, prob_vec in zip(ids, probs):
                ensemble_preds[img_id] += prob_vec

            total_models_used += 1

            # Free memory
            del model
            torch.cuda.empty_cache()

    # 4. Average and Save
    if total_models_used == 0:
        raise RuntimeError("No models were successfully loaded and used for inference.")

    logger.info(f"Averaging predictions from {total_models_used} models.")

    submission_rows = []
    # Iterate in the order of the test_df to maintain structure
    for img_id in test_df["image_id"].values:
        # Average the accumulated probabilities
        avg_probs = ensemble_preds[img_id] / total_models_used

        # Create row dictionary
        row = {"image_id": img_id}
        for idx, label in enumerate(Config.class_labels):
            row[label] = avg_probs[idx]
        submission_rows.append(row)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Ensure column order matches requirements: image_id, healthy, multiple_diseases, rust, scab
    cols = ["image_id"] + Config.class_labels
    submission_df = submission_df[cols]

    # Save to disk
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    submission_df.to_csv(Config.submission_path, index=False)
    logger.info(f"Submission saved to {Config.submission_path}")
