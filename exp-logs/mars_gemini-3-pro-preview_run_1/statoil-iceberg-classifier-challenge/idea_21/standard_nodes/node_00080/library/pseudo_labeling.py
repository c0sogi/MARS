import os
import numpy as np
import pandas as pd
import torch
from library.network import IcebergResNet
from library.inference import predict_with_tta
from library.config import WORK_DIR, SSL_CONF_HIGH, SSL_CONF_LOW, SSL_VAR_THRESH, SEED
from library.utils import set_seed


def generate_ensemble_stats(
    model_paths,
    test_loader,
    device,
    load_cached_data=True,
    cache_dir=WORK_DIR,
    cache_filename="teacher_predictions_stats.parquet",
):
    """
    Generates prediction statistics (mean and std) from a Teacher Ensemble on the test set.
    Uses caching to avoid re-running inference.

    Args:
        model_paths (list): List of paths to model checkpoints.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.
        load_cached_data (bool): Whether to try loading from cache.
        cache_dir (str): Directory to store the cache file.
        cache_filename (str): Name of the cache file.

    Returns:
        pd.DataFrame: DataFrame containing 'id', 'mean_prob', 'std_prob'.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_filename)

    # 1. Try Load from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading teacher ensemble stats from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating teacher ensemble predictions...")
    set_seed(SEED)  # Ensure deterministic behavior

    all_preds_dict = {}  # Key: ID, Value: List of probs

    # 2. Iterate through models and collect predictions
    for i, path in enumerate(model_paths):
        print(f"Processing Teacher Model {i+1}/{len(model_paths)}: {path}")

        if not os.path.exists(path):
            print(f"Warning: Model path {path} does not exist. Skipping.")
            continue

        # Initialize and load model
        model = IcebergResNet()
        model.to(device)

        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get(
            "model_state_dict", checkpoint.get("state_dict", checkpoint)
        )

        # Handle module prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)

        # Predict with TTA
        ids, probs = predict_with_tta(model, test_loader, device)

        # Store predictions
        for img_id, prob in zip(ids, probs):
            if img_id not in all_preds_dict:
                all_preds_dict[img_id] = []
            all_preds_dict[img_id].append(prob)

    # 3. Compute Statistics
    print("Computing ensemble statistics...")
    stats_data = []

    for img_id, probs_list in all_preds_dict.items():
        if not probs_list:
            continue

        probs_array = np.array(probs_list)
        mean_prob = np.mean(probs_array)
        std_prob = np.std(probs_array)

        stats_data.append(
            {
                "id": img_id,
                "mean_prob": mean_prob,
                "std_prob": std_prob,
                "num_votes": len(probs_list),
            }
        )

    df_stats = pd.DataFrame(stats_data)

    # 4. Save to Cache
    print(f"Saving teacher ensemble stats to {cache_path}")
    df_stats.to_parquet(cache_path, index=False)

    return df_stats


def filter_pseudo_labels(
    stats_df, conf_high=SSL_CONF_HIGH, conf_low=SSL_CONF_LOW, var_thresh=SSL_VAR_THRESH
):
    """
    Filters prediction statistics to select high-confidence, low-variance samples.

    Args:
        stats_df (pd.DataFrame): DataFrame with 'mean_prob' and 'std_prob'.
        conf_high (float): Threshold for positive class (Iceberg).
        conf_low (float): Threshold for negative class (Ship).
        var_thresh (float): Maximum allowed standard deviation.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'label' (0 or 1) for selected samples.
    """
    print(
        f"Filtering Pseudo-Labels (High>{conf_high}, Low<{conf_low}, Var<{var_thresh})..."
    )

    # Criteria 1: High Confidence
    # Criteria 2: Low Variance

    # Mask for Icebergs (Class 1)
    mask_iceberg = (stats_df["mean_prob"] > conf_high) & (
        stats_df["std_prob"] < var_thresh
    )

    # Mask for Ships (Class 0)
    mask_ship = (stats_df["mean_prob"] < conf_low) & (stats_df["std_prob"] < var_thresh)

    # Apply masks
    df_iceberg = stats_df[mask_iceberg].copy()
    df_iceberg["label"] = 1.0

    df_ship = stats_df[mask_ship].copy()
    df_ship["label"] = 0.0

    # Combine
    df_selected = pd.concat([df_iceberg, df_ship], ignore_index=True)

    print(f"Total Test Samples: {len(stats_df)}")
    print(f"Selected Pseudo-Labels: {len(df_selected)}")
    print(f"   - Icebergs: {len(df_iceberg)}")
    print(f"   - Ships: {len(df_ship)}")

    return df_selected[["id", "label"]]


def extract_pseudo_dataset(test_data_dict, pseudo_labels_df):
    """
    Extracts the image and angle data for the selected pseudo-labeled samples.

    Args:
        test_data_dict (dict): Dictionary containing 'images', 'angles', 'ids' of the full test set.
        pseudo_labels_df (pd.DataFrame): DataFrame with selected 'id' and 'label'.

    Returns:
        tuple: (images, angles, labels, ids) as numpy arrays.
    """
    print("Extracting pseudo-labeled data arrays...")

    # Create a lookup for the test data
    # Assuming test_data_dict['ids'] aligns with images/angles indices
    test_ids = test_data_dict["ids"]

    # Map ID to index in the test arrays
    id_to_idx = {uid: i for i, uid in enumerate(test_ids)}

    selected_indices = []
    selected_labels = []
    selected_ids = []

    # Iterate through selected pseudo labels
    for _, row in pseudo_labels_df.iterrows():
        uid = row["id"]
        label = row["label"]

        if uid in id_to_idx:
            idx = id_to_idx[uid]
            selected_indices.append(idx)
            selected_labels.append(label)
            selected_ids.append(uid)

    if not selected_indices:
        print("Warning: No matching IDs found in test data dictionary.")
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=object),
        )

    # Extract data using indices
    pseudo_images = test_data_dict["images"][selected_indices]
    pseudo_angles = test_data_dict["angles"][selected_indices]
    pseudo_labels = np.array(selected_labels, dtype=np.float32)
    pseudo_ids = np.array(selected_ids)

    print(f"Extracted {len(pseudo_images)} pseudo-labeled samples.")

    return pseudo_images, pseudo_angles, pseudo_labels, pseudo_ids
