import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import (
    N_FOLDS,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    SUBMISSION_PATH,
    TEST_META_PATH,
    get_model_path,
)
from library.utils import seed_everything, get_device, load_checkpoint
from library.dataset import load_data, IcebergDataset, get_transforms
from library.model import IcebergEfficientNet


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA).
    Averages predictions for original, horizontally flipped, and vertically flipped images.

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities (N, 1).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, angles in loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Original
            logits_orig = model(images, angles)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (Flip dimension 3: Width)
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h, angles)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip (Flip dimension 2: Height)
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v, angles)
            probs_v = torch.sigmoid(logits_v)

            # Average probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0
            all_preds.append(avg_probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def generate_submission():
    """
    Main function to generate the submission file.
    Ensembles predictions from all folds using TTA.
    """
    seed_everything(SEED)
    device = get_device()
    print(f"Inference device: {device}")

    # Load Test Data
    print("Loading test data...")
    images, angles, _ = load_data(mode="test", load_cached_data=True)

    # Create Dataset and Loader
    # We use 'valid' transforms which is just ToTensorV2, as TTA is handled manually
    test_dataset = IcebergDataset(
        images, angles, labels=None, transform=get_transforms(mode="valid")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize array to store ensemble predictions
    ensemble_preds = np.zeros((len(images), 1), dtype=np.float32)

    # Iterate over folds
    for fold_idx in range(N_FOLDS):
        print(f"Processing Fold {fold_idx + 1}/{N_FOLDS}...")

        # Initialize model
        model = IcebergEfficientNet()
        model.to(device)

        # Load weights
        model_path = get_model_path(fold_idx)
        try:
            load_checkpoint(model_path, model)
        except FileNotFoundError:
            print(
                f"Warning: Checkpoint for fold {fold_idx} not found at {model_path}. Skipping this fold."
            )
            continue

        # Predict with TTA
        preds = predict_with_tta(model, test_loader, device)
        ensemble_preds += preds

    # Average over folds
    final_preds = ensemble_preds / N_FOLDS

    # Prepare Submission
    print("Generating submission file...")

    # Load test metadata to get IDs
    df_test_meta = pd.read_csv(TEST_META_PATH)

    # Ensure alignment
    if len(df_test_meta) != len(final_preds):
        raise ValueError(
            f"Mismatch between metadata rows ({len(df_test_meta)}) and predictions ({len(final_preds)})"
        )

    # Create DataFrame
    submission = pd.DataFrame(
        {"id": df_test_meta["id"], "is_iceberg": final_preds.flatten()}
    )

    # Save
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
