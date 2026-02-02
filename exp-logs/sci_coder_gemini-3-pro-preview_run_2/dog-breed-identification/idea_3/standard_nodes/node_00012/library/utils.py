import os
import random
import numpy as np
import torch
from library.config import SEED, CACHE_DIR


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_embeddings(embeddings, labels, ids, prefix, folder=CACHE_DIR):
    """
    Saves embeddings, labels, and IDs to disk as .npy files.

    Args:
        embeddings (np.ndarray): The feature matrix.
        labels (np.ndarray or None): The target labels. Can be None for test set.
        ids (np.ndarray or None): The image IDs.
        prefix (str): Prefix for filenames (e.g., 'train_convnext').
        folder (str): Directory to save files. Defaults to CACHE_DIR.
    """
    os.makedirs(folder, exist_ok=True)

    # Save embeddings
    emb_path = os.path.join(folder, f"{prefix}_embeddings.npy")
    np.save(emb_path, embeddings)

    # Save labels if provided
    if labels is not None:
        lbl_path = os.path.join(folder, f"{prefix}_labels.npy")
        np.save(lbl_path, labels)

    # Save IDs if provided
    if ids is not None:
        ids_path = os.path.join(folder, f"{prefix}_ids.npy")
        np.save(ids_path, ids)


def load_embeddings(prefix, folder=CACHE_DIR):
    """
    Loads embeddings, labels, and IDs from disk.

    Args:
        prefix (str): Prefix for filenames (e.g., 'train_convnext').
        folder (str): Directory to load files from. Defaults to CACHE_DIR.

    Returns:
        tuple: (embeddings, labels, ids).
               Returns None if the embeddings file is missing.
               labels and ids will be None if their respective files are missing.
    """
    emb_path = os.path.join(folder, f"{prefix}_embeddings.npy")
    lbl_path = os.path.join(folder, f"{prefix}_labels.npy")
    ids_path = os.path.join(folder, f"{prefix}_ids.npy")

    # If embeddings don't exist, we can't proceed
    if not os.path.exists(emb_path):
        return None

    embeddings = np.load(emb_path)

    labels = None
    if os.path.exists(lbl_path):
        labels = np.load(lbl_path)

    ids = None
    if os.path.exists(ids_path):
        ids = np.load(ids_path)

    return embeddings, labels, ids
