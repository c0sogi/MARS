import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from library.config import Config
from library.data import load_data_to_memory, get_test_dataloader
from library.model import get_model
from library.utils import get_logger

# Initialize logger
logger = get_logger("inference")


def get_tta_transforms(images):
    """
    Generates 8 Dihedral variants (D4 Group) of the input batch.

    Args:
        images (torch.Tensor): Batch of images with shape (B, C, H, W).

    Returns:
        list[torch.Tensor]: List of 8 tensors representing the geometric variants.
    """
    variants = []

    # --- Base Rotations (0, 90, 180, 270) ---
    # k=0
    variants.append(images)
    # k=1 (90 degrees)
    variants.append(torch.rot90(images, k=1, dims=[2, 3]))
    # k=2 (180 degrees)
    variants.append(torch.rot90(images, k=2, dims=[2, 3]))
    # k=3 (270 degrees)
    variants.append(torch.rot90(images, k=3, dims=[2, 3]))

    # --- Flipped Rotations ---
    # Horizontal Flip
    hflip = torch.flip(images, dims=[3])

    # k=0 (flipped)
    variants.append(hflip)
    # k=1 (flipped + 90)
    variants.append(torch.rot90(hflip, k=1, dims=[2, 3]))
    # k=2 (flipped + 180)
    variants.append(torch.rot90(hflip, k=2, dims=[2, 3]))
    # k=3 (flipped + 270)
    variants.append(torch.rot90(hflip, k=3, dims=[2, 3]))

    return variants


def predict_batch_with_tta(model, images, device):
    """
    Predicts probabilities for a batch using 8-view Test Time Augmentation.

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of images.
        device (str): Device to run inference on.

    Returns:
        torch.Tensor: Averaged probabilities for the batch (B, 1).
    """
    model.eval()
    images = images.to(device)

    # Generate TTA variants
    variants = get_tta_transforms(images)

    batch_probs = []

    with torch.no_grad():
        for variant in variants:
            # Model forward pass
            # In inference mode, model returns averaged logits from MSD head (B, C)
            logits = model(variant)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            batch_probs.append(probs)

    # Stack predictions from all views: (8, B, C)
    stacked_probs = torch.stack(batch_probs)

    # Average across views: (B, C)
    avg_probs = stacked_probs.mean(dim=0)

    return avg_probs


def inference_single_fold(fold_idx, test_loader, device):
    """
    Performs inference on the test set using a specific fold's model.

    Args:
        fold_idx (int): The fold index to load.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Compute device.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    # Construct checkpoint path
    checkpoint_path = os.path.join(
        Config.checkpoints_dir, f"best_model_fold_{fold_idx}.pth"
    )

    if not os.path.exists(checkpoint_path):
        logger.warning(
            f"Checkpoint for Fold {fold_idx} not found at {checkpoint_path}. Skipping."
        )
        return None

    logger.info(f"Loading model for Fold {fold_idx}...")

    # Initialize model architecture
    model = get_model()
    model.to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    all_probs = []

    # Inference Loop
    for i, images in enumerate(test_loader):
        # Predict with TTA
        probs = predict_batch_with_tta(model, images, device)
        all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    return np.concatenate(all_probs)


def run_inference():
    """
    Main execution function for the inference module.
    Loads data, runs ensemble inference, and generates the submission file.
    """
    logger.info("Starting Inference Pipeline...")

    # 1. Load Test Data
    # load_data_to_memory handles caching logic. We ignore train data here.
    _, _, test_images, test_ids = load_data_to_memory(load_cached_data=True)

    # Handle Debug Mode (Slice IDs to match the DataLoader's slicing)
    if Config.debug:
        logger.info(
            f"Debug mode enabled. Slicing test set to {Config.debug_sample_size} samples."
        )
        test_ids = test_ids[: Config.debug_sample_size]
        # test_images is sliced inside get_test_dataloader

    # 2. Prepare DataLoader
    test_loader = get_test_dataloader(test_images)
    logger.info(f"Test DataLoader ready. Batches: {len(test_loader)}")

    # 3. Run Ensemble Inference
    fold_predictions = []

    for fold in range(Config.n_folds):
        preds = inference_single_fold(fold, test_loader, Config.device)
        if preds is not None:
            fold_predictions.append(preds)

    if not fold_predictions:
        logger.error("No valid predictions generated. Aborting submission.")
        return

    # 4. Aggregate Predictions
    # Stack predictions from all folds: (N_Folds, N_Samples, 1)
    stacked_preds = np.stack(fold_predictions)

    # Average across folds
    final_preds = np.mean(stacked_preds, axis=0)

    # Flatten to 1D array
    final_preds = final_preds.ravel()

    # Verify alignment
    if len(final_preds) != len(test_ids):
        logger.error(f"Shape Mismatch: Preds {len(final_preds)} vs IDs {len(test_ids)}")
        return

    # 5. Generate Submission File
    df_submission = pd.DataFrame({"id": test_ids, "label": final_preds})

    # Save to disk
    os.makedirs(Config.submission_dir, exist_ok=True)
    df_submission.to_csv(Config.submission_path, index=False)

    logger.info(f"Submission successfully saved to {Config.submission_path}")
    logger.info(f"Submission Head:\n{df_submission.head()}")


if __name__ == "__main__":
    # This block is included for standalone testing but will not be executed
    # by the main pipeline script if it imports run_inference.
    run_inference()
