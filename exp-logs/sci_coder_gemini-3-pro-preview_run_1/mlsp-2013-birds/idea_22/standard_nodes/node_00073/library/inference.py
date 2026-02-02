import os
import torch
import numpy as np
import pandas as pd
from library import config, data, utils


def predict_probs(model, loader, device, apply_flip=False):
    """
    Runs inference on a dataloader and returns probabilities.

    Args:
        model (torch.nn.Module): The model to use for inference.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device to run inference on.
        apply_flip (bool): If True, horizontally flips images before inference.

    Returns:
        np.ndarray: Array of probabilities with shape (N, num_classes).
    """
    model.eval()
    probs_list = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            if apply_flip:
                # Flip along the width dimension (dim 3 for N,C,H,W)
                images = torch.flip(images, dims=[3])

            logits = model(images)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu().numpy())

    return np.vstack(probs_list)


def predict_tta(model, device):
    """
    Performs Multi-Scale Test-Time Augmentation (TTA) using a single model.
    Iterates through config.TTA_WIDTHS, generating predictions for both original
    and horizontally flipped views, then averages them.

    Args:
        model (torch.nn.Module): The model to use.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Averaged probabilities (N, num_classes).
    """
    accumulated_probs = None
    num_views = 0

    # Iterate over the defined TTA widths
    for width in config.TTA_WIDTHS:
        # Get a DataLoader configured for this specific width
        # We ignore train/val loaders here
        _, _, test_loader = data.get_dataloaders(tta_width=width, load_cached_data=True)

        # 1. Original View
        probs_orig = predict_probs(model, test_loader, device, apply_flip=False)

        # 2. Flipped View
        probs_flip = predict_probs(model, test_loader, device, apply_flip=True)

        # Accumulate
        if accumulated_probs is None:
            accumulated_probs = probs_orig + probs_flip
        else:
            accumulated_probs += probs_orig + probs_flip

        num_views += 2

    # Average across all views (widths * 2 flips)
    final_probs = accumulated_probs / num_views
    return final_probs


def predict_ensemble_tta(models, device):
    """
    Performs Multi-Scale TTA using an ensemble of models (e.g., Teachers).
    Averages the TTA predictions of each model.

    Args:
        models (list of torch.nn.Module): List of loaded models.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Ensemble averaged probabilities.
    """
    ensemble_probs = None

    for model in models:
        model_probs = predict_tta(model, device)

        if ensemble_probs is None:
            ensemble_probs = model_probs
        else:
            ensemble_probs += model_probs

    return ensemble_probs / len(models)


def save_pseudo_labels(probs, output_path):
    """
    Saves soft predictions as pseudo-labels in Parquet format.

    Args:
        probs (np.ndarray): Probability matrix (N, num_classes).
        output_path (str): Path to save the parquet file.
    """
    # Load test metadata to ensure correct rec_id alignment
    # We use the cache if available to speed up loading
    df_test = data.load_metadata("test", load_cached_data=True)

    # Cite debug_lesson_4: Enforce Consistent Data Subsetting
    if config.DEBUG_MAX_SAMPLES is not None:
        df_test = df_test.iloc[: config.DEBUG_MAX_SAMPLES]

    rec_ids = df_test["rec_id"].values

    if len(rec_ids) != len(probs):
        raise ValueError(
            f"Mismatch between metadata rows ({len(rec_ids)}) and prediction rows ({len(probs)})"
        )

    # Construct DataFrame
    cols = [f"species_{i}" for i in range(config.NUM_CLASSES)]
    df_pseudo = pd.DataFrame(probs, columns=cols)
    df_pseudo.insert(0, "rec_id", rec_ids)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pseudo.to_parquet(output_path, index=False)
    print(f"Pseudo-labels saved to {output_path}")


def generate_submission(model, device, output_path):
    """
    Generates the final submission file using the Single-Scale inference strategy
    specified for the Student model.

    Args:
        model (torch.nn.Module): The final trained student model.
        device (str): Device to run inference on.
        output_path (str): Path to save the submission CSV.
    """
    # Use the standard inference width defined in config (single scale)
    _, _, test_loader = data.get_dataloaders(
        tta_width=config.IMG_WIDTH_TEST, load_cached_data=True
    )

    # Single forward pass (no flips, fixed resolution)
    probs = predict_probs(model, test_loader, device, apply_flip=False)

    # Load metadata for IDs
    df_test = data.load_metadata("test", load_cached_data=True)

    # Cite debug_lesson_4: Enforce Consistent Data Subsetting
    if config.DEBUG_MAX_SAMPLES is not None:
        df_test = df_test.iloc[: config.DEBUG_MAX_SAMPLES]

    rec_ids = df_test["rec_id"].values

    submission_rows = []

    # Format: Id,Probability
    # Id = rec_id * 100 + species_number
    for i, rid in enumerate(rec_ids):
        sample_probs = probs[i]
        for species_idx, p in enumerate(sample_probs):
            submission_rows.append(
                {"Id": int(rid * 100 + species_idx), "Probability": float(p)}
            )

    df_sub = pd.DataFrame(submission_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
