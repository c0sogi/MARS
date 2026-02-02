import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.data import get_dataloaders
from library.models import WhaleClassifier
from library.engine import predict
from library.utils import seed_everything


def infer_on_test(config, model_checkpoints, device, load_cached_preds=True):
    """
    Runs inference on the test set using an ensemble of models.

    Args:
        config (Config): Configuration object.
        model_checkpoints (list): List of tuples (model_name, checkpoint_path).
        device (torch.device): Device to run inference on.
        load_cached_preds (bool): If True, tries to load predictions from disk.

    Returns:
        np.array: Averaged probability predictions for the test set.
    """
    seed_everything(config.SEED)

    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(config.CACHE_DIR, "round1_ensemble_test_preds.npy")

    # 1. Try loading from cache
    if load_cached_preds and os.path.exists(cache_path):
        print(f"Loading cached test predictions from {cache_path}...")
        try:
            avg_preds = np.load(cache_path)
            # Verify shape matches test set size
            test_df = pd.read_csv(config.TEST_CSV)
            if len(avg_preds) == len(test_df):
                return avg_preds
            else:
                print("Cached predictions size mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Starting inference on test set with {len(model_checkpoints)} models...")

    # Get Test Loader
    # We use the data loader from library.data
    test_loader = get_dataloaders(config, mode="test", load_cached_data=True)

    all_model_preds = []

    for model_name, ckpt_path in model_checkpoints:
        print(f"Processing {model_name} from {ckpt_path}...")

        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
            continue

        # Initialize Model
        model = WhaleClassifier(
            model_name=model_name,
            pretrained=False,  # Weights loaded from checkpoint
            in_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
        )

        # Load Weights
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Handle state dict key mismatch if necessary (e.g. 'state_dict' wrapper)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )

        # Remove 'module.' prefix if trained with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()

        # Predict
        # predict() returns flattened numpy array of probabilities
        preds = predict(model, test_loader, device)
        all_model_preds.append(preds)

    if not all_model_preds:
        raise RuntimeError("No valid predictions generated. Check model paths.")

    # Average predictions
    avg_preds = np.mean(all_model_preds, axis=0)

    # Save to cache
    np.save(cache_path, avg_preds)
    print(f"Saved test predictions to {cache_path}")

    return avg_preds


def generate_pseudo_labels(config, test_probs, conf_high=None, conf_low=None):
    """
    Generates a DataFrame of pseudo-labels based on prediction confidence.

    Args:
        config (Config): Configuration object.
        test_probs (np.array): Probability predictions for the test set.
        conf_high (float, optional): Threshold for positive class (1).
        conf_low (float, optional): Threshold for negative class (0).

    Returns:
        pd.DataFrame: DataFrame containing 'clip' and 'label' for pseudo-labeled samples.
    """
    # Use config defaults if not provided
    if conf_high is None:
        conf_high = config.PSEUDO_LABEL_CONF_HIGH
    if conf_low is None:
        conf_low = config.PSEUDO_LABEL_CONF_LOW

    print(
        f"Generating pseudo-labels with thresholds: High>={conf_high}, Low<={conf_low}"
    )

    # Load Test Metadata to get clip names
    test_df = pd.read_csv(config.TEST_CSV)

    if len(test_df) != len(test_probs):
        raise ValueError(
            f"Shape mismatch: Test metadata has {len(test_df)} rows, predictions have {len(test_probs)}."
        )

    # Identify indices
    high_conf_indices = np.where(test_probs >= conf_high)[0]
    low_conf_indices = np.where(test_probs <= conf_low)[0]

    # Create Pseudo DataFrames
    # Positive Class (1)
    pos_df = test_df.iloc[high_conf_indices].copy()
    pos_df["label"] = 1

    # Negative Class (0)
    neg_df = test_df.iloc[low_conf_indices].copy()
    neg_df["label"] = 0

    # Combine
    pseudo_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Select only required columns
    # library.data.get_dataloaders expects 'clip' and 'label' to map back to file paths
    pseudo_df = pseudo_df[["clip", "label"]]

    # Logging
    n_pos = len(pos_df)
    n_neg = len(neg_df)
    total = len(test_df)
    n_pseudo = len(pseudo_df)

    print(f"Total Test Samples: {total}")
    print(f"Pseudo-Labels Generated: {n_pseudo} ({n_pseudo/total:.4f})")
    print(f"  - Class 1 (Whale) >={conf_high}: {n_pos}")
    print(f"  - Class 0 (Noise) <={conf_low}: {n_neg}")

    # Save for reference
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    save_path = os.path.join(config.WORKING_DIR, "pseudo_labels_summary.csv")
    pseudo_df.to_csv(save_path, index=False)
    print(f"Pseudo-label summary saved to {save_path}")

    return pseudo_df
