import os
import random
import numpy as np
import torch
import nltk
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein distance error rate.

    The score is the sum of Levenshtein distances between predicted and target
    sequences, divided by the total number of gestures in the target sequences.

    Args:
        predictions (list of list of int): Predicted gesture label sequences.
        targets (list of list of int): Ground truth gesture label sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_gestures = 0

    for pred, target in zip(predictions, targets):
        # Ensure inputs are lists
        p = list(pred) if pred is not None else []
        t = list(target) if target is not None else []

        # Calculate Levenshtein distance
        dist = nltk.edit_distance(p, t)

        total_distance += dist
        total_gestures += len(t)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures


def compute_class_weights():
    """
    Retrieves and converts the class weights from Config into a PyTorch tensor.

    Returns:
        torch.Tensor: A tensor containing class weights, moved to the configured device.
    """
    weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32)
    return weights.to(Config.get_device())


def collate_fn(batch):
    """
    Collate function for PyTorch DataLoader to handle variable-length sequences.

    Performs the following:
    1. Pads feature sequences to the maximum length in the batch.
    2. Generates a boolean mask indicating valid frames (True) vs padding (False).
    3. Pads frame-wise labels if present.
    4. Aggregates metadata like sample IDs and target sequences.

    Args:
        batch (list of dict): A list of data samples from the Dataset.

    Returns:
        dict: A dictionary containing:
            - 'features': Padded feature tensor (Batch, MaxLen, InputDim)
            - 'mask': Boolean mask tensor (Batch, MaxLen)
            - 'lengths': Tensor of original sequence lengths (Batch,)
            - 'frame_labels': Padded label tensor (Batch, MaxLen) [Optional]
            - 'sample_ids': List of sample IDs
            - 'target_sequence': List of target gesture sequences [Optional]
    """
    # Filter out any None items that might have resulted from loading errors
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    # Extract features and lengths
    features = [item["features"] for item in batch]
    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)

    # Pad features (Batch, Time, Dim)
    # Using 0.0 for padding value
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    # Create Boolean Mask (Batch, Time)
    # True indicates a valid frame, False indicates padding
    batch_size = padded_features.size(0)
    max_len = padded_features.size(1)

    # Create a range tensor (0, 1, ..., max_len-1) and expand to batch size
    # Compare with lengths to create mask
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)

    result = {
        "features": padded_features,
        "mask": mask,
        "lengths": lengths,
        "sample_ids": [item["sample_id"] for item in batch],
    }

    # Handle frame-wise labels for training/validation
    if "frame_labels" in batch[0]:
        labels = [item["frame_labels"] for item in batch]
        # Pad labels with 0 (Background class index)
        # The mask will ensure these padding positions are ignored in loss calculation
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
        result["frame_labels"] = padded_labels

    # Handle target sequences for validation metrics
    if "target_sequence" in batch[0]:
        result["target_sequence"] = [item["target_sequence"] for item in batch]

    return result
