import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_logger
from library.model_factory import create_model
from library.calibration import TemperatureScaler

logger = get_logger(name="inference")


def predict_with_tta(model, loader, device=Config.DEVICE):
    """
    Generates logits using Test-Time Augmentation (Horizontal Flip).
    Runs inference on both the original and horizontally flipped images,
    averaging the raw logits before softmax.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Computation device ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Averaged logits of shape (N, C).
    """
    model.eval()
    model.to(device)
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            # Handle different loader returns: (img, label) or (img, id)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)

            # Forward pass 1: Original
            logits_orig = model(images)

            # Forward pass 2: Horizontal Flip
            # Tensor shape: (B, C, H, W). Flip on W (dim 3)
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)

            # Average logits
            avg_logits = (logits_orig + logits_flip) / 2.0
            all_logits.append(avg_logits.cpu())

    return torch.cat(all_logits, dim=0)


def get_labels(loader):
    """
    Extracts labels from a DataLoader.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing labels.

    Returns:
        torch.Tensor: Tensor of all labels.
    """
    all_labels = []
    for batch in loader:
        # Assumes batch is (images, labels)
        if len(batch) >= 2:
            labels = batch[1]
            all_labels.append(labels)
    return torch.cat(all_labels, dim=0)


def generate_submission(train_loader, val_loader, test_loader, class_list):
    """
    Orchestrates the inference pipeline:
    1. Identifies trained models based on Config.
    2. Performs TTA inference on Validation set (for calibration) and Test set.
    3. Calibrates predictions using Temperature Scaling.
    4. Aggregates ensemble predictions.
    5. Saves the final submission file.

    Args:
        train_loader: Not used in inference but kept for signature consistency if needed.
        val_loader: DataLoader for validation set (used for calibration).
        test_loader: DataLoader for test set.
        class_list: List of class names.
    """
    logger.info("Starting Inference Pipeline...")

    ensemble_probs = []
    device = Config.DEVICE

    # Retrieve validation labels once for calibration
    if Config.USE_TEMP_SCALING:
        logger.info("Extracting validation labels for calibration...")
        val_labels = get_labels(val_loader)

    # Iterate over defined architectures
    for model_name in Config.MODEL_ARCHS:
        # Determine model path (prioritize SWA if enabled/exists, else Best)
        swa_path = os.path.join(Config.WORKING_DIR, f"{model_name}_swa.pth")
        best_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

        if Config.USE_SWA and os.path.exists(swa_path):
            model_path = swa_path
            logger.info(f"[{model_name}] Found SWA checkpoint: {model_path}")
        elif os.path.exists(best_path):
            model_path = best_path
            logger.info(f"[{model_name}] Found Best checkpoint: {model_path}")
        else:
            logger.warning(f"[{model_name}] No checkpoint found. Skipping.")
            continue

        # Load Model
        try:
            # num_classes matches the length of class_list
            model = create_model(
                model_name, num_classes=len(class_list), pretrained=False
            )
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
        except Exception as e:
            logger.error(f"[{model_name}] Error loading model: {e}")
            continue

        # Calibration Logic
        scaler = None
        if Config.USE_TEMP_SCALING:
            logger.info(
                f"[{model_name}] Running validation inference for calibration..."
            )
            val_logits = predict_with_tta(model, val_loader, device)

            scaler = TemperatureScaler()
            # Fit the scaler on validation logits/labels
            scaler.fit(val_logits, val_labels)

        # Test Inference
        logger.info(f"[{model_name}] Running test inference...")
        test_logits = predict_with_tta(model, test_loader, device)

        # Apply Calibration/Softmax
        if scaler:
            logger.info(f"[{model_name}] Applying temperature scaling...")
            probs = scaler.get_probabilities(test_logits)
        else:
            probs = torch.softmax(test_logits, dim=1)

        ensemble_probs.append(probs.numpy())

    if not ensemble_probs:
        logger.error("No predictions generated. Cannot create submission.")
        return

    # Ensemble Aggregation
    logger.info(f"Aggregating predictions from {len(ensemble_probs)} models...")
    avg_probs = np.mean(ensemble_probs, axis=0)

    # Generate Submission File
    logger.info("Preparing submission dataframe...")

    # Extract IDs from test loader
    test_ids = []
    for batch in test_loader:
        # Test loader yields (image, id)
        _, batch_ids = batch
        test_ids.extend(batch_ids)

    df_sub = pd.DataFrame(avg_probs, columns=class_list)
    df_sub.insert(0, "id", test_ids)

    # Save to ./submission/submission.csv
    output_dir = "./submission"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "submission.csv")

    df_sub.to_csv(output_path, index=False)
    logger.info(f"Submission saved successfully to {output_path}")
