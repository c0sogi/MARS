import os
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.model import IcebergResNet18
from library.data_loader import get_test_loader


def predict_with_tta(model, images, angles, device):
    """
    Predicts with Test Time Augmentation: Original, H-Flip, V-Flip.
    Returns average probability for the batch.

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of images (B, C, H, W).
        angles (torch.Tensor): Batch of incidence angles.
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Averaged probabilities for the batch.
    """
    model.eval()
    with torch.no_grad():
        # Move to device
        images = images.to(device)
        angles = angles.to(device)

        # 1. Original
        logits_orig = model(images, angles)
        probs_orig = torch.sigmoid(logits_orig)

        # 2. Horizontal Flip (dim 3 is width for NCHW)
        images_h = torch.flip(images, [3])
        logits_h = model(images_h, angles)
        probs_h = torch.sigmoid(logits_h)

        # 3. Vertical Flip (dim 2 is height for NCHW)
        images_v = torch.flip(images, [2])
        logits_v = model(images_v, angles)
        probs_v = torch.sigmoid(logits_v)

        # Average probabilities
        avg_probs = (probs_orig + probs_h + probs_v) / 3.0

    return avg_probs.cpu().numpy()


def generate_submission():
    """
    Generates the submission file using the single trained model.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission using device: {device}")

    # Get Test Loader
    test_loader, test_ids = get_test_loader(load_cached_data=True)

    # Load Model
    model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Cannot generate submission.")
        return

    print("Predicting with Best Model...")
    model = IcebergResNet18().to(device)
    model = load_checkpoint(model, model_path, device)

    # Inference loop
    all_preds = []
    for images, angles in test_loader:
        probs = predict_with_tta(model, images, angles, device)
        all_preds.append(probs)

    # Concatenate
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0).flatten()
    else:
        print("No predictions generated.")
        return

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": all_preds})

    # Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())
