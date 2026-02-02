import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library import utils, data


def generate_pseudo_labels(
    models, device=None, load_cached_data=False, cache_path=None
):
    """
    Generates pseudo-labels for the test set using the provided model(s).
    Applies Test-Time Augmentation (Horizontal Flip) and Ensemble Averaging.

    Args:
        models (list or nn.Module): Single model or list of trained PyTorch models.
        device (torch.device, optional): Device to run inference on. Defaults to Config.DEVICE.
        load_cached_data (bool): Whether to attempt loading results from cache.
        cache_path (str, optional): File path to save/load the pseudo-labels parquet file.

    Returns:
        pd.DataFrame: DataFrame containing 'rec_id' and predicted probabilities for each species.
    """
    if device is None:
        device = utils.get_device()

    # --- Caching Logic ---
    if load_cached_data and cache_path and os.path.exists(cache_path):
        try:
            # print(f"Loading cached pseudo-labels from {cache_path}")
            return pd.read_parquet(cache_path)
        except Exception:
            # If load fails, proceed to regeneration
            pass

    # --- Setup ---
    if not isinstance(models, list):
        models = [models]

    # Set all models to evaluation mode
    for model in models:
        model.to(device)
        model.eval()

    # Get Test Loader (using the standard data pipeline)
    # We ignore train/val loaders here
    _, _, test_loader = data.get_dataloaders()

    all_probs = []
    all_rec_ids = []

    # --- Inference Loop ---
    with torch.no_grad():
        for images, _, rec_ids in test_loader:
            images = images.to(device)

            # Test-Time Augmentation (TTA): Horizontal Flip
            # Images are (Batch, Channel, Height, Width). Flip on dim 3 (Width).
            images_flipped = torch.flip(images, dims=[3])

            batch_probs_sum = None

            for model in models:
                # 1. Forward pass - Original
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

                # 2. Forward pass - Flipped
                logits_flip = model(images_flipped)
                probs_flip = torch.sigmoid(logits_flip)

                # 3. Average TTA predictions for this model
                probs_avg = (probs_orig + probs_flip) / 2.0

                # Accumulate for Ensemble
                if batch_probs_sum is None:
                    batch_probs_sum = probs_avg
                else:
                    batch_probs_sum += probs_avg

            # Average across all models in the ensemble
            ensemble_probs = batch_probs_sum / len(models)

            all_probs.append(ensemble_probs.cpu().numpy())
            all_rec_ids.extend(rec_ids.numpy())

    # --- Post-Processing ---
    all_probs = np.concatenate(all_probs, axis=0)
    all_rec_ids = np.array(all_rec_ids)

    # Construct DataFrame
    data_dict = {"rec_id": all_rec_ids}
    for i in range(Config.NUM_CLASSES):
        data_dict[f"species_{i}"] = all_probs[:, i]

    df = pd.DataFrame(data_dict)

    # Sanitize: Ensure no NaNs (though rare with sigmoid)
    if df.isnull().values.any():
        df = df.fillna(0.0)

    # Ensure rec_id is integer
    df["rec_id"] = df["rec_id"].astype(int)

    # --- Save to Cache ---
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path)

    return df


def prepare_combined_dataset(pseudo_labels_df):
    """
    Validates and prepares the pseudo-labels DataFrame for merging.

    The actual merging logic resides in `library.data.get_dataloaders`, which accepts
    this DataFrame as an argument. This function ensures the DataFrame has the
    correct schema required by the data loader.

    Args:
        pseudo_labels_df (pd.DataFrame): The generated pseudo-labels.

    Returns:
        pd.DataFrame: The validated DataFrame ready for `get_dataloaders`.
    """
    required_cols = ["rec_id"] + [f"species_{i}" for i in range(Config.NUM_CLASSES)]

    # Validate Schema
    missing_cols = [col for col in required_cols if col not in pseudo_labels_df.columns]
    if missing_cols:
        raise ValueError(f"Pseudo-labels DataFrame is missing columns: {missing_cols}")

    # Ensure rec_id is int (crucial for merging with metadata)
    pseudo_labels_df["rec_id"] = pseudo_labels_df["rec_id"].astype(int)

    return pseudo_labels_df
