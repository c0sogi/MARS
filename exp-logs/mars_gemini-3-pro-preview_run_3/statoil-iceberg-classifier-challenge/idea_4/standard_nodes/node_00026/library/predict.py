import os
import numpy as np
import pandas as pd
import torch
from library.config import (
    N_FOLDS,
    WORKING_DIR,
    DEVICE,
    SUBMISSION_FILE,
    TEST_META_CSV,
    BATCH_SIZE,
    NUM_WORKERS,
)
from library.model import IcebergCNN
from library.data import get_dataloaders
from library.utils import set_seed


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    Computes probabilities for the original image, a horizontal flip, and a vertical flip,
    then averages them for a robust prediction.

    Args:
        model (nn.Module): The trained PyTorch model.
        loader (DataLoader): DataLoader for the test set.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        np.ndarray: A 1D numpy array of predicted probabilities.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for images, angles in loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Prediction on Original Image
            out_orig = model(images, angles)

            # 2. Prediction on Horizontally Flipped Image
            # Dimension 3 is width
            images_h = torch.flip(images, [3])
            out_h = model(images_h, angles)

            # 3. Prediction on Vertically Flipped Image
            # Dimension 2 is height
            images_v = torch.flip(images, [2])
            out_v = model(images_v, angles)

            # Average the probabilities from all views
            # out tensors are shape (Batch, 1)
            avg_pred = (out_orig + out_h + out_v) / 3.0

            preds_list.append(avg_pred.cpu().numpy())

    # Concatenate all batches along the 0-th dimension
    # Resulting shape: (N_samples, 1)
    return np.concatenate(preds_list, axis=0)


def generate_submission(load_cached_data=True):
    """
    Generates the submission file by ensembling models from all folds.
    Uses TTA for each model and averages the results.

    Args:
        load_cached_data (bool): Whether to load pre-processed .npy files or re-process raw data.
    """
    set_seed(42)
    print("Initializing submission generation...")

    # 1. Load Data
    # We retrieve the test_loader. train/val loaders are ignored.
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        load_cached_data=load_cached_data,
        debug=False,  # Always generate full submission
    )

    # Load Test IDs from metadata to ensure perfect alignment with predictions
    # get_dataloaders sorts data based on metadata indices, so we use the same source for IDs
    df_test_meta = pd.read_csv(TEST_META_CSV)
    test_ids = df_test_meta["id"].values
    num_samples = len(test_ids)

    print(f"Test samples: {num_samples}")

    # 2. Ensemble Loop
    # Accumulate predictions from all available fold models
    ensemble_preds = np.zeros((num_samples, 1), dtype=np.float32)
    folds_found = 0

    for fold_idx in range(N_FOLDS):
        fold_dir = os.path.join(WORKING_DIR, f"fold_{fold_idx}")
        model_path = os.path.join(fold_dir, "model_best.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for Fold {fold_idx} not found at {model_path}. Skipping."
            )
            continue

        print(f"Predicting with Fold {fold_idx} model...")

        # Initialize model architecture
        model = IcebergCNN().to(DEVICE)

        # Load trained weights
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

        # Generate predictions with TTA
        fold_preds = predict_with_tta(model, test_loader, DEVICE)

        # Accumulate
        ensemble_preds += fold_preds
        folds_found += 1

    if folds_found == 0:
        raise FileNotFoundError("No trained models found in working directory.")

    # 3. Average and Save
    # Average over the number of models used
    avg_preds = ensemble_preds / folds_found

    # Flatten to 1D array (N_samples,)
    avg_preds = avg_preds.flatten()

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")
    print("Submission head:")
    print(submission_df.head())
