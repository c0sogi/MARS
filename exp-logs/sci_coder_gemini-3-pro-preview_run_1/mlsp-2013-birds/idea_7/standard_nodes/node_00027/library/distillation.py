import os
import torch
import numpy as np
import pandas as pd
from library.config import Config


def generate_ensemble_pseudo_labels(
    teachers, test_loader, device, load_cached_data=True
):
    """
    Generates soft pseudo-labels for the test set by averaging predictions from an ensemble of teacher models.

    This function performs the following steps:
    1. Checks if a cached result exists and loads it if requested.
    2. If not cached, sets all teacher models to evaluation mode.
    3. Iterates through the test_loader, performing inference with each teacher.
    4. Applies Sigmoid activation to logits and averages the probabilities across teachers.
    5. Aggregates results into a DataFrame mapping 'rec_id' to soft labels.
    6. Caches the result to disk using Parquet format.

    Args:
        teachers (list): List of trained PyTorch models (nn.Module). Can be None/empty if loading from cache.
        test_loader (DataLoader): DataLoader for the unlabeled test set.
        device (str): Device to perform inference on (e.g., 'cuda' or 'cpu').
        load_cached_data (bool): If True, attempts to load predictions from cache. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing 'rec_id' and soft labels for each species (species_0 ... species_18).
    """
    # Define cache path within the working directory
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "ensemble_pseudo_labels.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached ensemble pseudo-labels from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute if not cached
    print("Generating ensemble pseudo-labels...")

    if not teachers:
        raise ValueError(
            "Teachers list is empty or None, and cached data was not found/requested."
        )

    # Set all teachers to eval mode
    for model in teachers:
        model.eval()
        model.to(device)

    all_rec_ids = []
    all_preds = []

    with torch.no_grad():
        for images, _, rec_ids in test_loader:
            images = images.to(device)

            # Store rec_ids (convert tensor to numpy)
            all_rec_ids.extend(rec_ids.numpy())

            batch_size = images.size(0)
            # Initialize accumulator for ensemble probabilities
            batch_preds = torch.zeros((batch_size, Config.NUM_CLASSES), device=device)

            # Aggregate predictions from all teachers
            for model in teachers:
                logits = model(images)
                probs = torch.sigmoid(logits)
                batch_preds += probs

            # Average the probabilities
            batch_preds /= len(teachers)

            all_preds.append(batch_preds.cpu().numpy())

    # Concatenate results from all batches
    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
    else:
        all_preds = np.empty((0, Config.NUM_CLASSES))

    # Create DataFrame
    # Columns: rec_id, species_0, species_1, ...
    cols = ["rec_id"] + [f"species_{i}" for i in range(Config.NUM_CLASSES)]

    # Combine rec_ids and preds
    # Note: np.column_stack will cast rec_id to float because preds are float
    data = np.column_stack((all_rec_ids, all_preds))

    df_pseudo = pd.DataFrame(data, columns=cols)

    # Cast rec_id back to int for consistency
    df_pseudo["rec_id"] = df_pseudo["rec_id"].astype(int)

    # 3. Save to cache
    print(f"Saving ensemble pseudo-labels to {cache_path}")
    df_pseudo.to_parquet(cache_path, index=False)

    return df_pseudo
