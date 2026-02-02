import os
import random
import numpy as np
import torch
import scipy.io
import pandas as pd
from typing import List, Any, Dict, Optional, Tuple, Callable
from library.config import Config


def seed_everything(seed: int = Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_mat_file(path: str) -> Optional[Any]:
    """
    Safely loads a .mat file using scipy.io.loadmat.
    Returns the mat object with squeeze_me=True and struct_as_record=False.
    """
    if not os.path.exists(path):
        return None
    try:
        # Load mat file, handling struct as objects
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        return mat
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """
    Computes the Levenshtein distance between two sequences of integers.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # Deletion
                    matrix[x, y - 1] + 1,  # Insertion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                )
    return matrix[size_x - 1, size_y - 1]


def compute_challenge_score(
    predictions: List[List[int]], ground_truths: List[List[int]]
) -> float:
    """
    Computes the challenge metric: Sum of Levenshtein distances divided by
    total number of gestures in ground truth.
    """
    total_distance = 0
    total_truth_gestures = 0

    for pred, truth in zip(predictions, ground_truths):
        dist = levenshtein_distance(pred, truth)
        total_distance += dist
        total_truth_gestures += len(truth)

    if total_truth_gestures == 0:
        return 0.0

    return total_distance / total_truth_gestures


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
):
    """
    Saves the model checkpoint.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(state, path)


def load_checkpoint(
    model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], path: str
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """
    Loads the model checkpoint. Returns model, optimizer, epoch, and loss.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return model, optimizer, epoch, loss


def save_features(data: Dict[str, np.ndarray], path: str):
    """
    Saves feature arrays to a compressed .npz file.
    Ensures directory exists.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez_compressed(path, **data)


def load_features(path: str) -> Dict[str, np.ndarray]:
    """
    Loads feature arrays from a .npz file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found at {path}")
    return dict(np.load(path, allow_pickle=True))


def load_or_compute(
    cache_path: str,
    compute_func: Callable[[], Dict[str, np.ndarray]],
    load_cached_data: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Caching logic wrapper.
    1. If load_cached_data is True and file exists, load it.
    2. Otherwise, run compute_func(), save result to cache_path, and return it.
    """
    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading cached data from {cache_path}")
            return load_features(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute
    # print(f"Computing data (Cache miss or force recompute)...")
    data = compute_func()

    # Save
    # print(f"Saving data to cache at {cache_path}")
    save_features(data, cache_path)

    return data


def format_submission_row(session_id: str, predicted_labels: List[int]) -> str:
    """
    Formats a single prediction row for the submission file.
    Format: SessionID,Label1,Label2,...
    """
    labels_str = ",".join(map(str, predicted_labels))
    return f"{session_id},{labels_str}"


def write_submission_file(predictions: List[Tuple[str, List[int]]], output_path: str):
    """
    Writes the full submission file.
    predictions: List of (session_id, list_of_gesture_ids)
    """
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_path, "w") as f:
        for session_id, labels in predictions:
            line = format_submission_row(session_id, labels)
            f.write(line + "\n")
