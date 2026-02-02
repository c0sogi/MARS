import os
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast

from library.config import Config
from library.dataset import get_dataloaders
from library.model import get_artwork_model
from library.utils import load_checkpoint, find_optimal_threshold


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (Horizontal Flip).

    Args:
        model: PyTorch model.
        loader: DataLoader.
        device: Device to run inference on.

    Returns:
        tuple: (predictions tensor, targets tensor)
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            # targets are loaded but might be placeholders for test set

            # 1. Forward pass original
            with autocast():
                output_orig = model(images)
                # 2. Forward pass flipped
                # NCHW format, flip width (dim 3)
                output_flip = model(torch.flip(images, dims=[3]))

            # Apply sigmoid to get probabilities
            probs_orig = torch.sigmoid(output_orig)
            probs_flip = torch.sigmoid(output_flip)

            # Average probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0

            all_preds.append(avg_probs.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return preds, targets


def optimize_threshold(model, val_loader, device):
    """
    Finds the optimal probability threshold using the validation set.

    Args:
        model: PyTorch model.
        val_loader: Validation DataLoader.
        device: Device.

    Returns:
        float: Optimal threshold.
    """
    print("Running inference on validation set to optimize threshold...")
    preds, targets = predict_with_tta(model, val_loader, device)

    # Use library utility to find best threshold
    best_threshold, best_f1 = find_optimal_threshold(preds, targets)

    print(f"Optimal Threshold: {best_threshold}")
    print(f"Validation Micro F1 at Optimal Threshold: {best_f1}")

    return best_threshold


def generate_submission(model, test_loader, threshold, device):
    """
    Generates the submission file for the test set.

    Args:
        model: PyTorch model.
        test_loader: Test DataLoader.
        threshold: Probability threshold for binarization.
        device: Device.
    """
    print(f"Generating predictions for test set with threshold {threshold}...")
    preds, _ = predict_with_tta(model, test_loader, device)

    # Binarize predictions
    preds_np = preds.numpy()
    binary_preds = (preds_np > threshold).astype(int)

    # Get Test IDs
    # The dataset dataframe contains the IDs
    test_ids = test_loader.dataset.df["id"].values

    submission_rows = []
    for idx, row in enumerate(binary_preds):
        # Find indices where value is 1
        predicted_indices = np.where(row == 1)[0]

        # Format as space-separated string
        if len(predicted_indices) > 0:
            pred_str = " ".join(map(str, predicted_indices))
        else:
            pred_str = ""  # Handle case with no predictions

        submission_rows.append({"id": test_ids[idx], "attribute_ids": pred_str})

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save
    os.makedirs(Config.submission_dir, exist_ok=True)
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")


def run_inference():
    """
    Main execution function for inference pipeline.
    """
    device = Config.device
    print(f"Initializing inference on device: {device}")

    # 1. Load Data
    # We load cached data if available for speed
    # We need val_loader for threshold optimization and test_loader for submission
    print("Loading dataloaders...")
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Load Model
    print("Constructing model...")
    # Pretrained weights are not needed as we load from checkpoint
    model = get_artwork_model(num_classes=Config.num_classes, pretrained=False)
    model = model.to(device)

    # 3. Load Weights
    checkpoint_path = Config.model_save_path
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        load_checkpoint(checkpoint_path, model, device=device)
    else:
        print(
            f"WARNING: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    # 4. Optimize Threshold
    best_threshold = optimize_threshold(model, val_loader, device)

    # 5. Generate Submission
    generate_submission(model, test_loader, best_threshold, device)

    print("Inference completed successfully.")
