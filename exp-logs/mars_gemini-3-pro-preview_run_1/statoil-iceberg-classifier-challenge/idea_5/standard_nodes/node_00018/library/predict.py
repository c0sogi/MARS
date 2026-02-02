import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.utils import get_device
from library.model import IcebergResNet
from library.data import process_split, IcebergDataset, get_transforms

# Constants
METADATA_DIR = "./metadata"
SUBMISSION_DIR = "./submission"


def predict_with_tta(model, loader, device):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).
    Averages predictions from: Original, Horizontal Flip, Vertical Flip.

    Args:
        model (nn.Module): The trained PyTorch model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        tuple: (ids, preds) where ids is a numpy array of image IDs and
               preds is a numpy array of predicted probabilities.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Original
            logits = model(images, angles)
            probs = torch.sigmoid(logits)

            # 2. Horizontal Flip (dim 3 is width for NCHW tensor)
            images_h = torch.flip(images, [3])
            logits_h = model(images_h, angles)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip (dim 2 is height for NCHW tensor)
            images_v = torch.flip(images, [2])
            logits_v = model(images_v, angles)
            probs_v = torch.sigmoid(logits_v)

            # Average probabilities
            avg_probs = (probs + probs_h + probs_v) / 3.0

            all_preds.extend(avg_probs.cpu().numpy().flatten())
            all_ids.extend(ids)

    return np.array(all_ids), np.array(all_preds)


def generate_ensemble_submission(n_folds, batch_size, model_dir, load_cached_data=True):
    """
    Loads trained models for each fold, generates predictions, averages them,
    and saves the submission file.

    Args:
        n_folds (int): Number of folds to use for the ensemble.
        batch_size (int): Batch size for inference.
        model_dir (str): Directory containing the saved model weights.
        load_cached_data (bool): Whether to use cached data files.
    """
    device = get_device()
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print("Preparing test data for inference...")
    # Load Test Data using the library function
    # This respects the caching logic defined in library/data.py
    test_imgs, test_angs, _, test_ids = process_split(
        os.path.join(METADATA_DIR, "test_metadata.csv"), "test", load_cached_data
    )

    # Create Dataset and Loader
    # We use the 'test' transform which only does ToTensor
    test_ds = IcebergDataset(
        test_imgs, test_angs, None, test_ids, transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    fold_predictions = []
    final_ids = None

    # Iterate over folds
    for fold_idx in range(n_folds):
        model_path = os.path.join(model_dir, f"model_fold_{fold_idx}_best.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold_idx} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model for Fold {fold_idx} from {model_path}...")

        # Initialize model architecture
        model = IcebergResNet()
        model.to(device)

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Generate predictions
        ids, preds = predict_with_tta(model, test_loader, device)
        fold_predictions.append(preds)

        # Store IDs from the first successful fold to ensure alignment
        if final_ids is None:
            final_ids = ids
        elif not np.array_equal(final_ids, ids):
            raise ValueError(f"ID mismatch in Fold {fold_idx} predictions.")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    if not fold_predictions:
        raise RuntimeError("No models were loaded. Cannot generate submission.")

    # Ensemble Aggregation (Arithmetic Mean)
    print(f"Aggregating predictions from {len(fold_predictions)} models...")
    avg_preds = np.mean(fold_predictions, axis=0)

    # Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    df_sub = pd.DataFrame({"id": final_ids, "is_iceberg": avg_preds})
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
