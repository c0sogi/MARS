import os
import numpy as np
import torch
import torch.nn.functional as F

import library.config as config
import library.model as model_lib


def load_model(seed, device):
    """
    Loads the trained CactusResNet model for a specific seed.

    Args:
        seed (int): The seed corresponding to the model checkpoint.
        device (torch.device): The device to load the model onto.

    Returns:
        nn.Module: The loaded model in evaluation mode.
    """
    model = model_lib.CactusResNet()
    model = model.to(device)

    # Construct path to the saved model
    model_path = os.path.join(config.WORKING_DIR, f"model_seed_{seed}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def _get_batch_probs(model, images):
    """
    Helper to compute probabilities from the model.
    """
    logits = model(images)
    return torch.sigmoid(logits)


def predict_tta(model, dataloader, device):
    """
    Generates predictions using Test Time Augmentation (TTA).
    TTA Strategy: Original + Horizontal Flip + Vertical Flip.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the test set.
        device (torch.device): Compute device.

    Returns:
        tuple: (ids, probabilities)
            - ids (np.ndarray): Array of image IDs.
            - probabilities (np.ndarray): Array of predicted probabilities (N, 1).
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch_data in dataloader:
            # Handle unpacking based on dataset return signature
            # Test dataset returns (image, id)
            if len(batch_data) == 2:
                images, ids = batch_data
            else:
                # Fallback if labels happen to be present (though unlikely for test)
                images, _, ids = batch_data

            images = images.to(device)

            # 1. Original Pass
            probs_orig = _get_batch_probs(model, images)

            # 2. Horizontal Flip Pass (dim 3 is width)
            images_h = torch.flip(images, dims=[3])
            probs_h = _get_batch_probs(model, images_h)

            # 3. Vertical Flip Pass (dim 2 is height)
            images_v = torch.flip(images, dims=[2])
            probs_v = _get_batch_probs(model, images_v)

            # Average TTA predictions
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            all_probs.append(avg_probs.cpu().numpy())
            all_ids.extend(ids)

    return np.array(all_ids), np.concatenate(all_probs, axis=0)


def predict_ensemble(seeds, dataloader, device):
    """
    Runs the full inference pipeline:
    1. Iterates over all seeds.
    2. Loads the corresponding model.
    3. Runs TTA inference.
    4. Averages predictions across all seeds.

    Args:
        seeds (list): List of seeds to use for the ensemble.
        dataloader (DataLoader): Test dataloader.
        device (torch.device): Compute device.

    Returns:
        tuple: (ids, final_probabilities)
    """
    accumulated_probs = None
    final_ids = None

    for i, seed in enumerate(seeds):
        print(f"Running inference for Seed {seed}...")

        # Load model
        model = load_model(seed, device)

        # Predict with TTA
        ids, probs = predict_tta(model, dataloader, device)

        # Initialize accumulator
        if accumulated_probs is None:
            accumulated_probs = probs
            final_ids = ids
        else:
            # Ensure ID alignment (Test loader should be deterministic, but good to check conceptually)
            if not np.array_equal(final_ids, ids):
                raise ValueError("Mismatch in Test IDs between seeds during ensemble.")
            accumulated_probs += probs

    # Average across seeds
    final_probs = accumulated_probs / len(seeds)

    return final_ids, final_probs
