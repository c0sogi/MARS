import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import load_checkpoint
from library.model import HierarchicalEfficientNet
from library.dataset import get_test_dataloader


def predict_tta(model, loader, device):
    """
    Performs inference on the test dataset using Test Time Augmentation (Horizontal Flip).
    It averages the logits from the original and flipped images for the species head.

    Args:
        model (torch.nn.Module): The trained hierarchical model.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to perform inference on.

    Returns:
        pd.DataFrame: DataFrame containing 'Id' and 'Predicted' columns.
    """
    model.eval()
    all_preds = []
    all_ids = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # 1. Forward pass with original images
            outputs_orig = model(images)
            logits_orig = outputs_orig["species"]

            # 2. Forward pass with horizontally flipped images (TTA)
            # Flip along the width dimension (dim 3 for N, C, H, W)
            images_flipped = torch.flip(images, dims=[3])
            outputs_flip = model(images_flipped)
            logits_flip = outputs_flip["species"]

            # 3. Average logits
            avg_logits = (logits_orig + logits_flip) / 2.0

            # 4. Get predictions (argmax)
            preds = torch.argmax(avg_logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_ids.extend(ids)

    # Create DataFrame
    df_submission = pd.DataFrame({"Id": all_ids, "Predicted": all_preds})

    return df_submission


def run_inference(
    checkpoint_path,
    output_path=None,
    batch_size=Config.STAGE2_BATCH_SIZE,
    image_size=Config.STAGE2_IMAGE_SIZE,
    device=None,
):
    """
    Orchestrates the inference process: loads model, runs TTA prediction, and saves submission.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        image_size (int): Image resolution for inference.
        device (str or torch.device): Device to use. Defaults to Config.DEVICE.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    print(f"Initializing inference on {device}...")

    # Initialize Model
    # We set pretrained=False because we are loading a specific checkpoint
    # and want to avoid unnecessary downloads or overwrites.
    model = HierarchicalEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=False,
        num_classes_species=Config.NUM_CLASSES_SPECIES,
        num_classes_genus=Config.NUM_CLASSES_GENUS,
        num_classes_family=Config.NUM_CLASSES_FAMILY,
    )
    model = model.to(device)

    # Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        load_checkpoint(checkpoint_path, model, device=device)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None

    # Get DataLoader
    print(
        f"Creating test dataloader (Image Size: {image_size}, Batch Size: {batch_size})..."
    )
    test_loader = get_test_dataloader(image_size, batch_size)

    # Run Prediction
    print("Running TTA Inference...")
    df_submission = predict_tta(model, test_loader, device)

    # Save Submission
    if output_path is None:
        output_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

    return df_submission
