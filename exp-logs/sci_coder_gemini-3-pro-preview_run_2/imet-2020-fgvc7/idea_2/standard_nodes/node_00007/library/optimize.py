import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score
from library.config import Config
from library.utils import seed_everything, get_device
from library.model import ArtworkConvNeXt
from library.dataset import get_dataloaders


def get_validation_predictions(load_cached_data=True):
    """
    Generates or loads validation predictions and targets.

    Args:
        load_cached_data (bool): If True, attempts to load from disk cache first.

    Returns:
        tuple: (val_logits, val_targets) as numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_logits_path = os.path.join(Config.WORKING_DIR, "val_logits.npy")
    cache_targets_path = os.path.join(Config.WORKING_DIR, "val_targets.npy")

    # 1. Try to load cached data
    if load_cached_data:
        if os.path.exists(cache_logits_path) and os.path.exists(cache_targets_path):
            print("Loading validation predictions from cache...")
            val_logits = np.load(cache_logits_path)
            val_targets = np.load(cache_targets_path)
            return val_logits, val_targets
        else:
            print("Cache not found. Generating predictions...")
    else:
        print("Ignoring cache. Generating predictions...")

    # 2. Compute from scratch
    device = get_device()
    seed_everything(Config.SEED)

    # Load Model
    model = ArtworkConvNeXt(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # Weights will be loaded from checkpoint
        num_classes=Config.NUM_CLASSES,
    )

    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {Config.MODEL_PATH}")
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_PATH}. Using random weights (expect poor performance)."
        )

    model.to(device)
    model.eval()

    # Get DataLoader
    # We use the validation loader from the library
    _, val_loader = get_dataloaders(
        debug=Config.DEBUG, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    all_logits = []
    all_targets = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            all_logits.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    val_logits = np.concatenate(all_logits, axis=0)
    val_targets = np.concatenate(all_targets, axis=0)

    # 3. Save to cache
    print(f"Saving validation predictions to {Config.WORKING_DIR}...")
    np.save(cache_logits_path, val_logits)
    np.save(cache_targets_path, val_targets)

    return val_logits, val_targets


def optimize_thresholds(val_logits, val_targets, num_steps=100):
    """
    Finds the single optimal global threshold that maximizes the Micro F1 score.
    Cite solution_lesson_node_00006: Do not optimize thresholds independently for Micro F1.

    Args:
        val_logits (np.array): Model output logits of shape (N, C).
        val_targets (np.array): Ground truth labels of shape (N, C).
        num_steps (int): Number of threshold steps to check between 0 and 1.

    Returns:
        np.array: Array of shape (C,) containing the optimal global threshold repeated.
    """
    print("Optimizing global threshold for Micro F1...")

    # Convert logits to probabilities
    val_probs = 1 / (1 + np.exp(-val_logits))

    # Define search space
    thresholds = np.linspace(0.05, 0.95, num_steps)

    best_f1 = -1
    best_th = 0.5

    # Grid search for global threshold
    for th in thresholds:
        # Apply threshold globally
        preds = (val_probs >= th).astype(int)

        # Calculate Micro F1
        score = f1_score(val_targets, preds, average="micro", zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_th = th

    print(f"Best Global Threshold: {best_th:.4f} with Micro F1: {best_f1:.4f}")

    # Return broadcasted threshold array for compatibility
    num_classes = val_targets.shape[1]
    return np.full(num_classes, best_th)


def evaluate_with_thresholds(val_logits, val_targets, thresholds):
    """
    Calculates the Micro F1 score using specific thresholds.

    Args:
        val_logits (np.array): Model logits.
        val_targets (np.array): Ground truth.
        thresholds (np.array): Threshold per class.

    Returns:
        float: Micro F1 score.
    """
    val_probs = 1 / (1 + np.exp(-val_logits))

    # Apply thresholds broadcasting
    predictions = (val_probs >= thresholds[None, :]).astype(int)

    micro_f1 = f1_score(val_targets, predictions, average="micro", zero_division=0)
    return micro_f1


def main_optimization_pipeline(load_cached_data=True):
    """
    Main function to run the optimization pipeline.
    """
    # 1. Get Validation Data
    val_logits, val_targets = get_validation_predictions(
        load_cached_data=load_cached_data
    )

    # 2. Calculate Baseline Score (Threshold 0.5)
    baseline_thresholds = np.full(val_targets.shape[1], 0.5)
    baseline_f1 = evaluate_with_thresholds(val_logits, val_targets, baseline_thresholds)
    print(f"Baseline Validation Micro F1 (Threshold 0.5): {baseline_f1}")

    # 3. Optimize Thresholds
    best_thresholds = optimize_thresholds(val_logits, val_targets)

    # 4. Calculate Optimized Score
    optimized_f1 = evaluate_with_thresholds(val_logits, val_targets, best_thresholds)
    print(f"Optimized Validation Micro F1: {optimized_f1}")

    # 5. Save Thresholds
    thresholds_path = os.path.join(Config.WORKING_DIR, "best_thresholds.npy")
    np.save(thresholds_path, best_thresholds)
    print(f"Saved optimized thresholds to {thresholds_path}")

    return best_thresholds
