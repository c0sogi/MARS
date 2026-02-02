import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast

from library.config import Config
from library.utils import get_logger, seed_everything
from library.dataset import get_dataloaders
from library.modeling import ArtworkClassifier

logger = get_logger("inference")


def get_val_predictions(model, loader, device, load_cached_data=True):
    """
    Generates or loads validation predictions (logits) and targets.
    Implements strict caching logic using .npy files.

    Args:
        model: The trained PyTorch model.
        loader: Validation DataLoader.
        device: Torch device.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (logits, targets) as numpy arrays.
    """
    logits_path = os.path.join(Config.WORKING_DIR, "val_logits.npy")
    targets_path = os.path.join(Config.WORKING_DIR, "val_targets.npy")

    # 1. Try to load cached data
    if (
        load_cached_data
        and os.path.exists(logits_path)
        and os.path.exists(targets_path)
    ):
        logger.info(f"Loading cached validation predictions from {Config.WORKING_DIR}")
        try:
            logits = np.load(logits_path)
            targets = np.load(targets_path)
            return logits, targets
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Regenerating.")

    # 2. Compute from scratch
    logger.info("Generating validation predictions...")
    model.eval()

    all_logits = []
    all_targets = []

    # Disable gradients for inference
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with autocast():
                # Standard validation inference (single view)
                batch_logits = model(images)

            all_logits.append(batch_logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    try:
        np.save(logits_path, logits)
        np.save(targets_path, targets)
        logger.info(f"Saved validation predictions to {Config.WORKING_DIR}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return logits, targets


def optimize_threshold(logits, targets):
    """
    Finds the global threshold that maximizes Micro F1 score.

    Args:
        logits (np.ndarray): Raw model outputs.
        targets (np.ndarray): Binary ground truth.

    Returns:
        float: The optimal threshold.
    """
    logger.info("Optimizing threshold...")
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid

    best_thresh = 0.5
    best_f1 = -1.0

    # Search range: 0.1 to 0.9 with step 0.01
    thresholds = np.arange(0.1, 0.9, 0.01)

    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        score = f1_score(targets, preds, average="micro")

        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    logger.info(f"Best Threshold: {best_thresh} | Best Micro F1: {best_f1}")
    return best_thresh


def predict_test_set(model, loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    TTA Strategy: Average probabilities of original and horizontally flipped images.

    Args:
        model: Trained PyTorch model.
        loader: Test DataLoader.
        device: Torch device.

    Returns:
        tuple: (probabilities, ids)
    """
    logger.info("Generating test predictions with TTA...")
    model.eval()

    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            ids = batch["id"]

            # TTA: Forward pass 1 (Original)
            with autocast():
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

                # TTA: Forward pass 2 (Flipped)
                # Flip width dimension (N, C, H, W) -> dim 3
                images_flipped = torch.flip(images, dims=[3])
                logits_flip = model(images_flipped)
                probs_flip = torch.sigmoid(logits_flip)

            # Average probabilities
            probs_avg = (probs_orig + probs_flip) / 2.0

            all_probs.append(probs_avg.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_probs), all_ids


def create_submission(probs, ids, threshold, output_path):
    """
    Applies threshold and saves submission CSV.

    Args:
        probs (np.ndarray): Predicted probabilities.
        ids (list): List of image IDs.
        threshold (float): Decision threshold.
        output_path (str): Path to save CSV.
    """
    logger.info(
        f"Creating submission file at {output_path} with threshold {threshold}..."
    )

    # Binarize predictions
    preds = (probs > threshold).astype(int)

    submission_rows = []
    for i, img_id in enumerate(ids):
        # Get indices of positive classes
        indices = np.where(preds[i] == 1)[0]
        # Format as space-separated string
        attr_str = " ".join(map(str, indices))
        submission_rows.append({"id": img_id, "attribute_ids": attr_str})

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(output_path, index=False)
    logger.info("Submission saved.")


def run_inference(
    checkpoint_path=Config.STUDENT_CHECKPOINT,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.VAL_BATCH_SIZE,
    debug=Config.DEBUG,
):
    """
    Main inference pipeline entry point.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs on a subset of data.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 1. Load Model
    logger.info(f"Loading model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = ArtworkClassifier(Config.STUDENT_MODEL_NAME, Config.NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Get DataLoaders
    # Note: We pass batch_size as val_batch_size to get_dataloaders
    dataloaders = get_dataloaders(
        batch_size=Config.BATCH_SIZE,  # Unused for val/test
        val_batch_size=batch_size,
        debug=debug,
    )
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # 3. Optimize Threshold on Validation Set
    val_logits, val_targets = get_val_predictions(
        model, val_loader, device, load_cached_data=True
    )
    best_thresh = optimize_threshold(val_logits, val_targets)

    # 4. Predict on Test Set (with TTA)
    test_probs, test_ids = predict_test_set(model, test_loader, device)

    # 5. Generate Submission
    create_submission(test_probs, test_ids, best_thresh, output_path)
