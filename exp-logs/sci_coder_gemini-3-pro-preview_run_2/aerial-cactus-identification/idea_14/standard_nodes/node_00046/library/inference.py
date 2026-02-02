import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import CactusDataset
from library.model import HybridNarrowSEResNet
from library.utils import seed_everything


def load_ensemble_models(device):
    """
    Loads all trained models for the ensemble into memory.

    Args:
        device (torch.device): The device to load models onto.

    Returns:
        list: A list of loaded HybridNarrowSEResNet models.
    """
    models = []
    for seed in Config.SEEDS:
        model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

        # Initialize model architecture
        model = HybridNarrowSEResNet()

        # Load weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
            print(f"Loaded model for seed {seed}")
        else:
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )

    if not models:
        raise RuntimeError("No trained models found. Cannot proceed with inference.")

    return models


def predict_batch_ensemble(models, images):
    """
    Computes the averaged prediction for a batch of images across all models
    and TTA variations (Original, H-Flip, V-Flip).

    Args:
        models (list): List of loaded models.
        images (torch.Tensor): Batch of images (B, C, H, W).

    Returns:
        np.array: Averaged probabilities for the batch (B,).
    """
    # Prepare TTA versions
    images_h = torch.flip(images, dims=[3])  # Horizontal flip
    images_v = torch.flip(images, dims=[2])  # Vertical flip

    # Accumulator for probabilities
    total_probs = torch.zeros(images.size(0), 1, device=images.device)

    # Count total predictions per image (Models * Variations)
    # Variations = Original + H-Flip + V-Flip = 3
    num_variations = 3
    count = len(models) * num_variations

    with torch.no_grad():
        for model in models:
            # Original
            out_orig = model(images)
            total_probs += torch.sigmoid(out_orig)

            # H-Flip
            out_h = model(images_h)
            total_probs += torch.sigmoid(out_h)

            # V-Flip
            out_v = model(images_v)
            total_probs += torch.sigmoid(out_v)

    # Compute mean
    avg_probs = total_probs / count
    return avg_probs.cpu().numpy().flatten()


def run_inference():
    """
    Main inference routine.
    Loads data, loads models, computes predictions with TTA and Ensembling,
    and saves the submission file.
    """
    # 1. Setup
    seed_everything(42)  # Ensure deterministic behavior for dataloader shuffling if any
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 2. Load Data
    # We use the test metadata.
    # Note: load_cached_data=True allows using cached .npy files if available,
    # otherwise it processes from scratch using the metadata file.
    test_dataset = CactusDataset(
        metadata_path=Config.TEST_METADATA_PATH, phase="test", load_cached_data=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # 3. Load Models
    models = load_ensemble_models(device)
    print(f"Ensemble size: {len(models)} models")

    # 4. Inference Loop
    all_ids = []
    all_probs = []

    print("Starting prediction loop...")
    for images, _, ids in test_loader:
        images = images.to(device)

        # Get averaged predictions for this batch
        batch_probs = predict_batch_ensemble(models, images)

        all_ids.extend(ids)
        all_probs.extend(batch_probs)

    # 5. Save Submission
    submission_df = pd.DataFrame({"id": all_ids, "has_cactus": all_probs})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
