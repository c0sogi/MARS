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
    Generates the submission file using the trained ensemble.
    Loads models for all bags, performs TTA inference, averages results,
    and saves to submission.csv.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission using device: {device}")

    # Get Test Loader
    # load_cached_data=True ensures we use the preprocessed .npy files if available
    test_loader, test_ids = get_test_loader(load_cached_data=True)

    # Initialize list to store predictions from all bags
    # Each element will be a numpy array of shape (N_test,)
    all_bag_preds = []

    # Iterate through each bag in the ensemble
    for bag_idx in range(Config.NUM_BAGS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_bag_{bag_idx}.pth")

        if not os.path.exists(model_path):
            print(f"Model for Bag {bag_idx} not found at {model_path}. Skipping.")
            continue

        print(f"Predicting with Bag {bag_idx}...")

        # Initialize Model
        model = IcebergResNet18().to(device)

        # Load Weights
        try:
            model = load_checkpoint(model, model_path, device)
        except Exception as e:
            print(f"Failed to load model for Bag {bag_idx}: {e}")
            continue

        # Inference loop for this bag
        bag_preds = []
        for images, angles in test_loader:
            probs = predict_with_tta(model, images, angles, device)
            bag_preds.append(probs)

        # Concatenate batches for this bag to get shape (N_test, 1) or (N_test,)
        if len(bag_preds) > 0:
            bag_preds = np.concatenate(bag_preds, axis=0).flatten()
            all_bag_preds.append(bag_preds)

    # Average across bags
    if not all_bag_preds:
        print("No predictions generated. Please check if models are trained.")
        return

    # Stack to shape (Num_Bags, Num_Test_Samples)
    all_bag_preds = np.vstack(all_bag_preds)

    # Calculate mean across the ensemble dimension (axis 0)
    avg_preds = np.mean(all_bag_preds, axis=0)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())
