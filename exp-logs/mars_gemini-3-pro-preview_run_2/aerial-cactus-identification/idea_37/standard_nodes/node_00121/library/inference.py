import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model_components import UltraWideSERepNeXt
from library.dataset import get_loaders


def predict_with_tta(model, images, device):
    """
    Generates predictions for a batch of images using Test Time Augmentation (TTA).

    TTA Strategy:
    1. Original Image
    2. Horizontal Flip
    3. Vertical Flip

    Args:
        model (nn.Module): The trained model (in deploy mode).
        images (torch.Tensor): Batch of images (B, C, H, W).
        device (torch.device): Computation device.

    Returns:
        torch.Tensor: Averaged probabilities for the batch (B,).
    """
    # Ensure images are on the correct device
    images = images.to(device)

    # 1. Create Views
    img_orig = images
    img_hflip = torch.flip(images, dims=[3])  # Horizontal flip
    img_vflip = torch.flip(images, dims=[2])  # Vertical flip

    # 2. Forward Pass
    # The model outputs logits, so we apply sigmoid here
    logit_orig = model(img_orig)
    logit_hflip = model(img_hflip)
    logit_vflip = model(img_vflip)

    prob_orig = torch.sigmoid(logit_orig).view(-1)
    prob_hflip = torch.sigmoid(logit_hflip).view(-1)
    prob_vflip = torch.sigmoid(logit_vflip).view(-1)

    # 3. Aggregate (Arithmetic Mean)
    avg_prob = (prob_orig + prob_hflip + prob_vflip) / 3.0

    return avg_prob


def generate_submission():
    """
    Main inference routine.

    Steps:
    1. Loads the test dataset.
    2. Loads all trained model checkpoints defined in Config.SEEDS.
    3. Converts models to deployment mode (Structural Re-parameterization).
    4. Iterates through the test set, applying TTA and ensembling predictions.
    5. Saves the results to the submission CSV.
    """
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    # We only need the test loader
    _, _, test_loader = get_loaders(load_cached_data=True)

    # 2. Load Models
    models = []
    print(f"Loading models for seeds: {Config.SEEDS}")

    for seed in Config.SEEDS:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_seed_{seed}.pth")

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint not found for seed {seed} at {checkpoint_path}. Skipping."
            )
            continue

        # Initialize model in training topology to match saved state_dict
        model = UltraWideSERepNeXt(num_classes=Config.NUM_CLASSES, deploy=False)

        # Load weights
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading checkpoint for seed {seed}: {e}")
            continue

        model.to(device)
        model.eval()

        # 3. Structural Re-parameterization
        # Fuse multi-branch blocks into single convolutions for faster inference
        model.switch_to_deploy()

        models.append(model)

    if not models:
        print("Error: No valid models loaded. Cannot generate submission.")
        return

    print(f"Successfully loaded {len(models)} models. Starting inference...")

    results = {}

    # 4. Inference Loop
    with torch.no_grad():
        for images, ids in test_loader:
            # Accumulate probabilities across all models
            batch_ensemble_preds = torch.zeros(images.size(0), device=device)

            for model in models:
                # Get TTA averaged prediction for this model
                model_preds = predict_with_tta(model, images, device)
                batch_ensemble_preds += model_preds

            # Average across the ensemble
            batch_ensemble_preds /= len(models)

            # Move to CPU and store
            preds_np = batch_ensemble_preds.cpu().numpy()

            for img_id, pred in zip(ids, preds_np):
                results[img_id] = pred

    # 5. Save Submission
    submission_df = pd.DataFrame(
        [{"id": img_id, "has_cactus": prob} for img_id, prob in results.items()]
    )

    # Sort by ID to ensure consistent order
    submission_df = submission_df.sort_values("id")

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(submission_df)}")
