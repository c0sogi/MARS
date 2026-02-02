import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import UltraWideSERepNeXt
from library.dataset import CactusDataset
from library.utils import load_checkpoint


def load_ensemble(device):
    """
    Loads the ensemble of trained models.
    Applies structural re-parameterization (switch_to_deploy) to each model
    for optimized inference.

    Args:
        device (torch.device): The device to load models onto.

    Returns:
        list: A list of loaded and optimized UltraWideSERepNeXt models.
    """
    models = []
    for seed in Config.SEEDS:
        model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")

        # Initialize architecture
        model = UltraWideSERepNeXt()

        # Load weights
        # We wrap this in a try-except block to handle cases where a specific seed
        # might not have finished training during development/debugging.
        try:
            load_checkpoint(model_path, model, device=device)
            print(f"Loaded model for Seed {seed} from {model_path}")
        except FileNotFoundError:
            print(
                f"Warning: Model for Seed {seed} not found at {model_path}. Skipping."
            )
            continue

        # Optimize for inference: Fuse branches (RepVGG style)
        model.switch_to_deploy()

        model.to(device)
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError("No models were loaded. Cannot proceed with inference.")

    return models


def predict_with_tta(models, images):
    """
    Predicts probabilities for a batch of images using the ensemble and Test Time Augmentation.
    TTA Strategy: Average of [Original, Horizontal Flip, Vertical Flip].

    Args:
        models (list): List of trained models.
        images (torch.Tensor): Batch of images (B, C, H, W).

    Returns:
        np.ndarray: Averaged probabilities for the batch.
    """
    # Prepare TTA views
    # 1. Original
    # 2. Horizontal Flip (dim 3 is width)
    # 3. Vertical Flip (dim 2 is height)

    views = [images, torch.flip(images, dims=[3]), torch.flip(images, dims=[2])]

    total_probs = None
    count = 0

    with torch.no_grad():
        for model in models:
            for view in views:
                # Forward pass
                logits = model(view)
                probs = torch.sigmoid(logits)

                if total_probs is None:
                    total_probs = probs
                else:
                    total_probs += probs

                count += 1

    # Average across all models and all views
    avg_probs = total_probs / count
    return avg_probs.cpu().numpy().flatten()


def generate_submission(debug=False):
    """
    Generates the submission file for the test set.

    Args:
        debug (bool): If True, processes only a small subset of the test data.
    """
    print("Starting submission generation...")

    device = Config.DEVICE

    # 1. Load Ensemble
    models = load_ensemble(device)
    print(f"Ensemble size: {len(models)} models.")

    # 2. Prepare Data
    test_dataset = CactusDataset(mode="test", load_cached_data=True)

    if debug:
        print("Debug mode: processing first 100 images only.")
        indices = list(range(min(len(test_dataset), 100)))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Inference Loop
    all_ids = []
    all_probs = []

    print("Running inference with TTA...")
    for images, ids in test_loader:
        images = images.to(device)

        # Get averaged predictions from ensemble + TTA
        probs = predict_with_tta(models, images)

        all_ids.extend(ids)
        all_probs.extend(probs)

    # 4. Create Submission DataFrame
    df_submission = pd.DataFrame({"id": all_ids, "has_cactus": all_probs})

    # 5. Save
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_submission.to_csv(save_path, index=False)
    print(f"Submission saved to: {save_path}")
    print(f"Total predictions: {len(df_submission)}")

    # Print head for verification
    print("Submission head:")
    print(df_submission.head())
