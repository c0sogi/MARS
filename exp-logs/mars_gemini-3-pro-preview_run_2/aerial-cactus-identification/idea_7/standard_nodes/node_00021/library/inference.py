import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import CactusDataset, get_transforms
from library.model import CactusResUNet


def predict_with_tta(model, loader, device):
    """
    Performs inference using Test Time Augmentation (TTA).
    Strategy: Average predictions of Original, Horizontal Flip, and Vertical Flip.

    Args:
        model (nn.Module): The trained neural network.
        loader (DataLoader): DataLoader for the test set.
        device (torch.device): Compute device.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            # 1. Original Image
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (dim 3 is Width)
            images_hflip = torch.flip(images, dims=[3])
            logits_hflip = model(images_hflip)
            probs_hflip = torch.sigmoid(logits_hflip)

            # 3. Vertical Flip (dim 2 is Height)
            images_vflip = torch.flip(images, dims=[2])
            logits_vflip = model(images_vflip)
            probs_vflip = torch.sigmoid(logits_vflip)

            # Average probabilities across augmentations
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

            all_probs.append(avg_probs.cpu().numpy())

    # Concatenate all batches and flatten to 1D array
    return np.concatenate(all_probs).flatten()


def generate_submission(test_data):
    """
    Generates the submission file by ensembling predictions from multiple models.
    Applies TTA for each model and averages the results across all seeds.

    Args:
        test_data (tuple): A tuple containing (test_images, test_ids).
    """
    test_images, test_ids = test_data

    # Ensure output directories exist
    Config.setup()
    device = torch.device(Config.DEVICE)

    # Prepare Test DataLoader
    # Note: labels are None for inference
    test_dataset = CactusDataset(
        images=test_images, labels=None, transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Starting inference on {len(test_images)} images...")

    # Accumulator for ensemble predictions
    ensemble_probs = np.zeros(len(test_images), dtype=np.float64)
    successful_models = 0

    # Iterate over all seeds to load models and predict
    for seed in Config.SEEDS:
        model_path = Config.get_model_path(seed)

        if not os.path.exists(model_path):
            print(
                f"Warning: Model checkpoint not found for Seed {seed} at {model_path}. Skipping."
            )
            continue

        print(f"Processing Seed {seed}...")

        # Initialize and load model
        model = CactusResUNet().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)

        # Predict with TTA
        probs = predict_with_tta(model, test_loader, device)

        # Accumulate
        ensemble_probs += probs
        successful_models += 1

    if successful_models == 0:
        raise RuntimeError(
            "No models were successfully loaded. Cannot generate submission."
        )

    # Compute final average
    final_probs = ensemble_probs / successful_models

    # Create submission DataFrame
    df_submission = pd.DataFrame({"id": test_ids, "has_cactus": final_probs})

    # Save to CSV
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")
