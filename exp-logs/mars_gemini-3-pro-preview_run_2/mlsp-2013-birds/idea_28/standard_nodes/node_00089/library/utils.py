import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic cuDNN algorithms.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for consistent distillation results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pos_weight(df_train, device):
    """
    Calculates positive weights for BCEWithLogitsLoss based on class imbalance in the training set.
    Formula: pos_weight = number_of_negatives / number_of_positives

    Args:
        df_train (pd.DataFrame): Training dataframe containing label columns (prefixed with 'species_').
        device (torch.device): The device to place the resulting weight tensor on.

    Returns:
        torch.Tensor: A tensor of weights for each class.
    """
    # Identify label columns based on metadata convention
    label_cols = [c for c in df_train.columns if c.startswith("species_")]

    # Calculate counts
    num_samples = len(df_train)
    pos_counts = df_train[label_cols].sum().values
    neg_counts = num_samples - pos_counts

    # Avoid division by zero by clamping positive counts to at least 1
    # (EDA indicates all classes have samples, but this is a safety measure)
    pos_counts = np.maximum(pos_counts, 1)

    # Calculate weights
    weights = neg_counts / pos_counts

    return torch.tensor(weights, dtype=torch.float32).to(device)


def save_oof_preds(preds, ids, save_path):
    """
    Saves Out-Of-Fold (OOF) predictions to a Parquet file for use in subsequent generations.

    Args:
        preds (np.ndarray or torch.Tensor): Prediction probabilities (shape: [N, num_classes]).
        ids (list, np.ndarray, or torch.Tensor): Corresponding recording IDs.
        save_path (str): Destination path for the Parquet file.
    """
    # Convert Tensors to Numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create DataFrame
    # Column names match the 'pred_X' convention expected by load_oof_preds
    num_classes = preds.shape[1]
    cols = [f"pred_{i}" for i in range(num_classes)]

    df = pd.DataFrame(preds, columns=cols)
    df["rec_id"] = ids

    # Save to Parquet (efficient storage, no pickle)
    df.to_parquet(save_path, index=False)


def load_oof_preds(load_path, ids):
    """
    Loads OOF predictions from a Parquet file and aligns them with the requested IDs.
    This is critical for ensuring the soft targets match the input data during training.

    Args:
        load_path (str): Path to the Parquet file containing OOF predictions.
        ids (list or np.ndarray): List of recording IDs to retrieve predictions for.

    Returns:
        np.ndarray: Aligned prediction probabilities corresponding to the input ids.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"OOF predictions file not found at {load_path}")

    # Read the data
    df = pd.read_parquet(load_path)

    # Set rec_id as index for efficient lookup
    df = df.set_index("rec_id")

    # Identify prediction columns
    pred_cols = [c for c in df.columns if c.startswith("pred_")]

    # Reindex the dataframe to match the order of the requested ids.
    # This aligns the soft targets with the current batch/dataset.
    df_aligned = df.reindex(ids)

    # Check for missing IDs (should not happen in a correct pipeline)
    if df_aligned.isnull().any().any():
        print(
            f"Warning: Missing OOF predictions for some IDs in {load_path}. Filling missing values with 0."
        )
        df_aligned = df_aligned.fillna(0)

    return df_aligned[pred_cols].values
