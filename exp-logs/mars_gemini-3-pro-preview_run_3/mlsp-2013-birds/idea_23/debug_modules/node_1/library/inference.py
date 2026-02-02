import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.data import BirdDataset, get_transforms
from library.models import get_model
from library.utils import save_formatted_submission


def apply_tta_shifts(images, shift_ratio=0.1):
    """
    Applies Horizontal Shift TTA: Original, Left Shift, Right Shift.
    Uses zero-padding for shifts to preserve temporal causality/structure.

    Args:
        images (torch.Tensor): Batch of images (B, C, H, W).
        shift_ratio (float): Fraction of width to shift.

    Returns:
        tuple: (original, left_shifted, right_shifted)
    """
    B, C, H, W = images.shape
    shift_pixels = int(W * shift_ratio)

    # Original
    orig = images

    # Left Shift: Move content left, pad right with 0
    # Content: images[:, :, :, shift:]
    # Pad: zeros at the end
    left_content = images[:, :, :, shift_pixels:]
    left_pad = torch.zeros(
        (B, C, H, shift_pixels), device=images.device, dtype=images.dtype
    )
    left_shifted = torch.cat([left_content, left_pad], dim=3)

    # Right Shift: Move content right, pad left with 0
    # Content: images[:, :, :, :-shift]
    # Pad: zeros at the start
    right_content = images[:, :, :, :-shift_pixels]
    right_pad = torch.zeros(
        (B, C, H, shift_pixels), device=images.device, dtype=images.dtype
    )
    right_shifted = torch.cat([right_pad, right_content], dim=3)

    return orig, left_shifted, right_shifted


def predict_with_tta(model, images, device):
    """
    Performs inference with TTA (Original + Left + Right) and averages probabilities.

    Args:
        model (torch.nn.Module): The trained model.
        images (torch.Tensor): Batch of input images.
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Averaged probability batch.
    """
    model.eval()
    with torch.no_grad():
        # Get TTA versions
        img_orig, img_left, img_right = apply_tta_shifts(images, shift_ratio=0.1)

        # Forward pass
        logits_orig = model(img_orig)
        logits_left = model(img_left)
        logits_right = model(img_right)

        # Apply Sigmoid to get probabilities
        probs_orig = torch.sigmoid(logits_orig)
        probs_left = torch.sigmoid(logits_left)
        probs_right = torch.sigmoid(logits_right)

        # Average probabilities
        avg_probs = (probs_orig + probs_left + probs_right) / 3.0

    return avg_probs


def generate_ensemble_predictions(config, test_loader, device):
    """
    Iterates through all models, folds, and checkpoints to generate ensemble predictions.
    Aggregates predictions by averaging probabilities.

    Args:
        config (Config): Configuration object.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Computation device.

    Returns:
        tuple: (list of rec_ids, numpy array of averaged probabilities)
    """
    num_samples = len(test_loader.dataset)
    num_classes = config.NUM_CLASSES

    # Accumulator for probabilities (N_samples, N_classes)
    final_accumulated_probs = np.zeros((num_samples, num_classes), dtype=np.float64)
    model_count = 0

    # Extract rec_ids once to ensure alignment
    rec_ids_list = []
    for _, rec_ids in test_loader:
        rec_ids_list.extend(rec_ids.numpy())

    # Iterate over defined architectures
    for model_name in config.ARCHITECTURES:
        checkpoint_base_dir = os.path.join(config.OUTPUT_DIR, "checkpoints", model_name)

        # Iterate over folds
        for fold_idx in range(config.N_FOLDS):
            # Find all checkpoints for this fold (Top-K are saved by Trainer)
            # Pattern: {model_name}_fold_{fold_idx}_epoch_*.pth
            search_pattern = os.path.join(
                checkpoint_base_dir, f"{model_name}_fold_{fold_idx}_*.pth"
            )
            checkpoints = glob.glob(search_pattern)

            if not checkpoints:
                print(f"Warning: No checkpoints found for {model_name} fold {fold_idx}")
                continue

            print(
                f"Processing {model_name} Fold {fold_idx}: Found {len(checkpoints)} checkpoints."
            )

            # Iterate over Snapshot Checkpoints
            for ckpt_path in checkpoints:
                # Initialize Model
                model = get_model(model_name, config, device=device)

                # Load Weights
                try:
                    checkpoint = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                except Exception as e:
                    print(f"Error loading {ckpt_path}: {e}")
                    continue

                # Inference Loop
                fold_probs = []
                with torch.no_grad():
                    for images, _ in test_loader:
                        images = images.to(device)
                        batch_probs = predict_with_tta(model, images, device)
                        fold_probs.append(batch_probs.cpu().numpy())

                # Concatenate batches for the full dataset
                fold_probs = np.concatenate(fold_probs, axis=0)

                # Accumulate
                final_accumulated_probs += fold_probs
                model_count += 1

                # Cleanup to save memory
                del model
                torch.cuda.empty_cache()

    if model_count == 0:
        raise RuntimeError("No models were successfully loaded and executed!")

    # Compute final average
    avg_probs = final_accumulated_probs / model_count

    return rec_ids_list, avg_probs


def run_inference():
    """
    Main entry point for the inference module.
    Loads data, runs the ensemble prediction pipeline, and saves the submission file.
    """
    # 1. Setup
    config = Config()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting Inference on device: {device}")

    # 2. Load Test Metadata
    test_csv_path = os.path.join(config.METADATA_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)

    # 3. Prepare Dataset and DataLoader
    # transforms="test" applies normalization and tensor conversion
    test_dataset = BirdDataset(
        test_df, config, transforms=get_transforms("test", config), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Generate Predictions
    rec_ids, probabilities = generate_ensemble_predictions(config, test_loader, device)

    # 5. Save Submission
    submission_path = "./submission/submission.csv"
    save_formatted_submission(rec_ids, probabilities, submission_path)

    print(f"Submission saved to {submission_path}")
    print("Inference Complete.")
