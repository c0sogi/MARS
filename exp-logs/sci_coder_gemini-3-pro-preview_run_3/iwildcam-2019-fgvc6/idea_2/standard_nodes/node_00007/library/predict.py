import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import AnimalClassifier
from library.dataset import get_dataloaders
from library.utils import load_checkpoint, seed_everything


def predict_tta(model, loader, device):
    """
    Performs inference on the test set using Test-Time Augmentation (TTA).
    Specifically, it averages predictions from the original image and a horizontally flipped version.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): The test data loader.
        device (torch.device): The device to run inference on.

    Returns:
        tuple: A tuple containing two lists: (image_ids, predicted_labels).
    """
    model.eval()
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for images, _, image_ids in loader:
            images = images.to(device)

            # 1. Forward pass with original images
            outputs_orig = model(images)

            # 2. Forward pass with horizontally flipped images (TTA)
            # Input shape is (N, C, H, W), so we flip dimension 3 (Width)
            images_flipped = torch.flip(images, dims=[3])
            outputs_flip = model(images_flipped)

            # 3. Average logits (Soft Voting)
            outputs_avg = (outputs_orig + outputs_flip) / 2.0

            # 4. Get final predictions
            preds = torch.argmax(outputs_avg, dim=1).cpu().numpy()

            all_ids.extend(image_ids)
            all_preds.extend(preds)

    return all_ids, all_preds


def generate_submission(
    checkpoint_path=Config.MODEL_CHECKPOINT_PATH, output_path=Config.SUBMISSION_PATH
):
    """
    Orchestrates the submission generation process: loads data, loads model,
    runs TTA inference, and saves the results to CSV.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        output_path (str): Path where the submission CSV should be saved.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # Load DataLoaders
    # We only need the test loader
    print("Loading DataLoaders...")
    _, _, test_loader = get_dataloaders()

    # Initialize Model
    # We use pretrained=False here because we are about to load specific weights
    # from our checkpoint, avoiding unnecessary downloads.
    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = AnimalClassifier(pretrained=False)
    model = model.to(device)

    # Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = load_checkpoint(checkpoint_path, model)

    if checkpoint is None:
        print(
            f"Warning: Checkpoint file {checkpoint_path} not found. Predictions will be based on random weights."
        )
    else:
        best_f1 = checkpoint.get("best_val_f1", "N/A")
        print(
            f"Checkpoint loaded successfully (Epoch: {checkpoint.get('epoch', 'N/A')}, Best Val F1: {best_f1})"
        )

    # Run Inference
    print("Running inference with Test-Time Augmentation...")
    ids, preds = predict_tta(model, test_loader, device)

    # Create Submission DataFrame
    df = pd.DataFrame({"Id": ids, "Predicted": preds})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
