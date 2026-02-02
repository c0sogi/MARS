import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

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
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1,  # Insertion
                )
    return matrix[size_x - 1, size_y - 1]


def compute_levenshtein_score(predictions, ground_truths):
    """
    Computes the normalized Levenshtein distance score.

    Args:
        predictions (list of list of int): Predicted gesture sequences.
        ground_truths (list of list of int): Ground truth gesture sequences.

    Returns:
        float: The average error rate (Total Distance / Total GT Gestures).
    """
    total_distance = 0
    total_len = 0

    for p, t in zip(predictions, ground_truths):
        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_len += len(t)

    if total_len == 0:
        return 0.0

    return total_distance / total_len


def generate_gaussian_boundary_targets(labels, num_frames, sigma=Config.BOUNDARY_SIGMA):
    """
    Generates Gaussian-smoothed boundary targets based on label transitions.

    Args:
        labels (list or np.array): Frame-wise labels (0 for background, 1-20 for gestures).
        num_frames (int): Total number of frames.
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        np.array: A 1D array of shape (num_frames,) containing soft boundary probabilities.
    """
    boundary_target = np.zeros(num_frames, dtype=np.float32)

    # Identify transition indices
    # A transition occurs at index i if label[i] != label[i-1]
    labels_arr = np.array(labels)
    if len(labels_arr) < 2:
        return boundary_target

    # Differences
    diff = labels_arr[1:] != labels_arr[:-1]
    transition_indices = np.where(diff)[0] + 1  # +1 because diff is shifted

    # Apply Gaussian kernel at each transition
    # Kernel window: +/- 3*sigma
    window_radius = int(3 * sigma)

    for idx in transition_indices:
        start = max(0, idx - window_radius)
        end = min(num_frames, idx + window_radius + 1)

        # Create grid for Gaussian
        x = np.arange(start, end)
        gaussian = np.exp(-0.5 * ((x - idx) / sigma) ** 2)

        # Accumulate (max to keep it as probability-like, though sum is also valid for density)
        # Using max to prevent values > 1.0 when transitions are close
        boundary_target[start:end] = np.maximum(boundary_target[start:end], gaussian)

    return boundary_target


def apply_median_filter(predictions, kernel_size=Config.MEDIAN_FILTER_K):
    """
    Applies a median filter to smooth discrete class predictions.

    Args:
        predictions (np.array): Array of shape (T,) or (B, T) containing class indices.
        kernel_size (int): Size of the median filter window (must be odd).

    Returns:
        np.array: Smoothed predictions.
    """
    from scipy.signal import medfilt

    if kernel_size % 2 == 0:
        kernel_size += 1

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    if predictions.ndim == 1:
        return medfilt(predictions, kernel_size=kernel_size).astype(int)
    elif predictions.ndim == 2:
        smoothed = []
        for seq in predictions:
            smoothed.append(medfilt(seq, kernel_size=kernel_size))
        return np.array(smoothed).astype(int)
    return predictions


def create_padding_mask(lengths, max_len=None):
    """
    Creates a boolean mask where True indicates a valid position and False indicates padding.

    Args:
        lengths (torch.Tensor): Tensor containing sequence lengths.
        max_len (int, optional): Maximum sequence length. If None, max(lengths) is used.

    Returns:
        torch.Tensor: Boolean mask of shape (Batch, Max_Len).
    """
    if max_len is None:
        max_len = lengths.max().item()

    batch_size = lengths.size(0)
    # Create range [0, 1, ..., max_len-1]
    ids = (
        torch.arange(0, max_len, device=lengths.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    # Mask is True where index < length
    mask = ids < lengths.unsqueeze(1)
    return mask


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.

    Args:
        batch: List of tuples (features, frame_labels, boundary_targets, sample_id)

    Returns:
        dict: Batch dictionary with padded tensors and masks.
    """
    # Unpack batch
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    boundaries_list = [item[2] for item in batch]
    ids_list = [item[3] for item in batch]

    # Get lengths
    lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)

    # Pad sequences
    # features: (T, Input_Dim) -> (B, Max_T, Input_Dim)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    # labels: (T,) -> (B, Max_T)
    # Use -100 or 0 for padding? Usually CrossEntropy ignores -100.
    # However, our mask will handle it. Let's use 0 (background) or a specific ignore index.
    # Given we use explicit masking in loss, 0 is safe if handled, but -1 is safer for debugging.
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-1)

    # boundaries: (T,) -> (B, Max_T)
    boundaries_padded = pad_sequence(
        boundaries_list, batch_first=True, padding_value=0.0
    )

    # Create mask
    mask = create_padding_mask(lengths, features_padded.size(1))

    return {
        "features": features_padded,
        "labels": labels_padded,
        "boundaries": boundaries_padded,
        "mask": mask,
        "lengths": lengths,
        "sample_ids": ids_list,
    }
