import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders
from library.model import build_model


def inference_with_tta(model, loader, device, classes):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): The device to run inference on.
        classes (list): List of class names corresponding to the model outputs.

    Returns:
        pd.DataFrame: A DataFrame containing the 'id' and probability columns for each class.
    """
    model.eval()
    results = []
    ids_list = []

    print("Starting inference with TTA (Original + Horizontal Flip)...")

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # 1. Original Image
            outputs_orig = model(images)
            probs_orig = F.softmax(outputs_orig, dim=1)

            # 2. Flipped Image (Horizontal Flip)
            # Input is (B, C, H, W), so we flip dimension 3 (Width)
            images_flipped = torch.flip(images, dims=[3])
            outputs_flipped = model(images_flipped)
            probs_flipped = F.softmax(outputs_flipped, dim=1)

            # Average probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0

            results.append(avg_probs.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate all batch results
    final_probs = np.concatenate(results, axis=0)

    # Create DataFrame
    df_sub = pd.DataFrame(final_probs, columns=classes)
    df_sub.insert(0, "id", ids_list)

    return df_sub


def generate_submission(
    checkpoint_path=Config.best_model_path,
    output_path=Config.submission_path,
    debug=Config.debug,
):
    """
    Orchestrates the submission generation process: loads data, builds model,
    loads weights, runs inference, and saves the result.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
        debug (bool): If True, runs on a subset of data (handled by Config/Dataset).
    """
    # 1. Setup
    set_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"Running inference on device: {device}")

    # 2. Data Loading
    # get_dataloaders returns (train, val, test, classes)
    print("Loading data...")
    _, _, test_loader, classes = get_dataloaders(load_cached_data=True)

    print(f"Number of test images: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(classes)}")

    # 3. Model Construction
    # We set pretrained=False because we are about to load our own trained weights
    model = build_model(num_classes=Config.num_classes, pretrained=False)
    model = model.to(device)

    # 4. Load Weights
    print(f"Loading model weights from {checkpoint_path}...")
    try:
        load_checkpoint(model, checkpoint_path)
    except FileNotFoundError:
        print(
            f"Error: Checkpoint not found at {checkpoint_path}. Ensure training has completed."
        )
        return

    # 5. Generate Predictions
    df_submission = inference_with_tta(model, test_loader, device, classes)

    # 6. Save Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving submission to {output_path}...")
    df_submission.to_csv(output_path, index=False)
    print("Submission generation completed.")
