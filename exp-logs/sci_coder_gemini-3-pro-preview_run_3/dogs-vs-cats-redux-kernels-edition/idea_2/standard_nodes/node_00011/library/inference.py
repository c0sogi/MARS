import os
import torch
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import get_logger
from library.dataset import get_dataloaders
from library.model import get_model

logger = get_logger("inference")


def predict_with_tta(model, loader, device, use_tta=False):
    """
    Generates predictions for the test set, optionally using Test Time Augmentation.

    Args:
        model: PyTorch model.
        loader: DataLoader for test data.
        device: 'cuda' or 'cpu'.
        use_tta (bool): Whether to use Test Time Augmentation (Horizontal Flip).

    Returns:
        tuple: (ids, probabilities)
    """
    model.eval()
    all_ids = []
    all_probs = []

    logger.info(f"Starting prediction loop (TTA={use_tta})...")

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # Forward pass 1: Original
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Extract probability for Class 1 (Dog)
            dog_probs = probs[:, 1]

            if use_tta:
                # Forward pass 2: Horizontal Flip
                # Flip along width dimension (dim 3: N, C, H, W)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.softmax(outputs_flipped, dim=1)

                # Extract probability for Class 1 (Dog)
                dog_probs_flipped = probs_flipped[:, 1]

                # Average probabilities
                dog_probs = (dog_probs + dog_probs_flipped) / 2.0

            # Move to CPU and store
            all_ids.extend(ids.numpy())
            all_probs.extend(dog_probs.cpu().numpy())

    return all_ids, all_probs


def create_submission(ids, probs, output_path=Config.SUBMISSION_PATH):
    """
    Formats predictions into a DataFrame and saves to CSV.

    Args:
        ids (list): List of image IDs.
        probs (list): List of probabilities for the positive class.
        output_path (str): Path to save the CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id": ids, "label": probs})

    # Sort by ID to ensure consistent order
    df = df.sort_values("id")

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")


def run_inference(
    checkpoint_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
    output_path=Config.SUBMISSION_PATH,
    device=Config.DEVICE,
    use_tta=Config.USE_TTA,
    batch_size=Config.BATCH_SIZE,
):
    """
    Main entry point to run inference pipeline.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
        device (str): Device to run inference on.
        use_tta (bool): Enable Test Time Augmentation.
        batch_size (int): Batch size for the dataloader.
    """
    # 1. Load Data
    # We unpack the tuple to get only the test_loader
    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    # 2. Load Model Architecture
    # We set pretrained=False because we are about to load our own fine-tuned weights.
    # This avoids downloading ImageNet weights unnecessarily.
    model = get_model(pretrained=False, device=device)

    # 3. Load Weights
    if not os.path.exists(checkpoint_path):
        logger.error(
            f"Checkpoint not found at {checkpoint_path}. Cannot run inference."
        )
        return

    logger.info(f"Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 4. Predict
    ids, probs = predict_with_tta(model, test_loader, device, use_tta=use_tta)

    # 5. Save Submission
    create_submission(ids, probs, output_path=output_path)
