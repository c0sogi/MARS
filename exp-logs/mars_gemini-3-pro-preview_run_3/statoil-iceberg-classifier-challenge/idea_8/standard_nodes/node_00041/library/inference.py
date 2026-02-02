import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import SimpleCNN
from library.data_loader import get_test_loader
from library.utils import set_seed


def predict_with_tta(model, test_loader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    Predicts on original and horizontally flipped images, then averages the probabilities.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Prediction on original images
            logits_orig = model(images, angles)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Prediction on horizontally flipped images
            # Images are (B, C, H, W). Horizontal flip is on the last dimension (W).
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip, angles)
            probs_flip = torch.sigmoid(logits_flip)

            # Average the probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0

            # Move to CPU and store
            all_probs.append(avg_probs.cpu().numpy())

    # Concatenate all batches
    return np.concatenate(all_probs).flatten()


def create_submission(load_cached_data=True):
    """
    Loads models from all folds, generates predictions with TTA, ensembles them,
    and creates the submission file.

    Args:
        load_cached_data (bool): Whether to use cached pre-processed data.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Loading test data...")
    test_loader, test_ids = get_test_loader(load_cached_data=load_cached_data)

    # Array to accumulate predictions from all folds
    # Initialize with zeros. Length matches the number of test samples.
    # We can determine length from ids or by running one pass, but ids is safest.
    num_test_samples = len(test_ids)
    ensemble_probs = np.zeros(num_test_samples, dtype=np.float64)

    folds_processed = 0

    print(f"Starting inference on {Config.N_FOLDS} folds...")

    for fold in range(Config.N_FOLDS):
        fold_model_path = os.path.join(
            Config.WORKING_DIR, f"fold_{fold}", "model_best.pth"
        )

        if not os.path.exists(fold_model_path):
            print(
                f"Warning: Model for fold {fold} not found at {fold_model_path}. Skipping."
            )
            continue

        print(f"Processing Fold {fold}...")

        # Initialize model architecture
        model = SimpleCNN()
        model.to(device)

        # Load weights
        checkpoint = torch.load(fold_model_path, map_location=device)
        # Handle cases where checkpoint is a dict vs just state_dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        # Generate predictions
        fold_probs = predict_with_tta(model, test_loader, device)

        # Accumulate
        ensemble_probs += fold_probs
        folds_processed += 1

    if folds_processed == 0:
        raise RuntimeError("No models were found to generate predictions.")

    # Average predictions
    avg_probs = ensemble_probs / folds_processed

    # Create submission DataFrame
    print("Creating submission file...")
    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_probs})

    # Save to CSV
    # Ensure directory exists (Config.setup() handles this, but good practice)
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False, float_format="%.6f")
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
