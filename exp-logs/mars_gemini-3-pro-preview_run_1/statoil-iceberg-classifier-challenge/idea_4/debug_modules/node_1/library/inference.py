import os
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.utils import get_device
from library.model import IcebergResNet34
from library.data_loader import get_test_loader


def predict_with_tta(model, loader, device):
    """
    Performs inference with Test Time Augmentation (Original + HFlip + VFlip).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The test data loader.
        device (torch.device): The computation device.

    Returns:
        tuple: (predictions numpy array, list of ids)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, angles, img_ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Original
            logits_orig = model(images, angles)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (flip width dimension, dim 3)
            images_hflip = torch.flip(images, dims=[3])
            logits_hflip = model(images_hflip, angles)
            probs_hflip = torch.sigmoid(logits_hflip)

            # 3. Vertical Flip (flip height dimension, dim 2)
            images_vflip = torch.flip(images, dims=[2])
            logits_vflip = model(images_vflip, angles)
            probs_vflip = torch.sigmoid(logits_vflip)

            # Average probabilities
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

            all_preds.append(avg_probs.cpu().numpy())
            all_ids.extend(img_ids)

    return np.concatenate(all_preds).flatten(), all_ids


def generate_ensemble_predictions(model_paths):
    """
    Generates submission file by averaging predictions from all fold models.

    Args:
        model_paths (list): List of file paths to the saved model checkpoints.
    """
    print("Starting Ensemble Prediction...")
    device = get_device()

    # Load test data (utilizing cache if available)
    # This ensures we use the same preprocessed data as training or previous runs
    test_loader = get_test_loader(load_cached_data=True)

    ensemble_preds = None
    test_ids = None

    for i, path in enumerate(model_paths):
        print(f"Processing model {i + 1}/{len(model_paths)}: {path}")

        # Instantiate Model
        model = IcebergResNet34()

        # Load weights
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at {path}")

        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)

        # Predict with TTA
        preds, ids = predict_with_tta(model, test_loader, device)

        if ensemble_preds is None:
            ensemble_preds = preds
            test_ids = ids
        else:
            # Verify IDs alignment (sanity check)
            if test_ids != ids:
                raise ValueError("Mismatch in test IDs between models.")
            ensemble_preds += preds

    if ensemble_preds is None:
        raise ValueError("No predictions generated. Check model_paths.")

    # Average over folds
    ensemble_preds /= len(model_paths)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": ensemble_preds})

    # Save
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving submission to {save_path}")
    df_sub.to_csv(save_path, index=False)
    print("Submission generation complete.")
