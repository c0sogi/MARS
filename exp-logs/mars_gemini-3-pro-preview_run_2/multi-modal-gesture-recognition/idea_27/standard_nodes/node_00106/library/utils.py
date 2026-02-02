import os
import random
import numpy as np
import torch
import nltk
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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sequence_mask(lengths, max_len=None):
    """
    Computes a boolean mask for a batch of variable-length sequences.

    Args:
        lengths (torch.Tensor): A 1D tensor containing the lengths of each sequence in the batch.
        max_len (int, optional): The maximum length to pad to. If None, uses the max value in lengths.

    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, max_len) where True indicates
                      valid positions and False indicates padding.
    """
    if max_len is None:
        max_len = lengths.max().item()

    batch_size = lengths.size(0)
    # Create a range tensor [0, 1, ..., max_len-1]
    # Expand it to (batch_size, max_len)
    range_tensor = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    )
    # Expand lengths to (batch_size, max_len)
    lengths_expanded = lengths.unsqueeze(1).expand(batch_size, max_len)

    # Mask is True where index < length
    mask = range_tensor < lengths_expanded
    return mask


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the model training state to a file.

    Args:
        state (dict): Dictionary containing model state, optimizer state, epoch, etc.
        filename (str): Name of the file to save.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)
    # print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filename="checkpoint.pth", map_location=None):
    """
    Loads a model checkpoint from a file.

    Args:
        filename (str): Name of the file to load.
        map_location (str or torch.device, optional): Device to map the storage to.

    Returns:
        dict: The loaded state dictionary.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    return torch.load(filepath, map_location=map_location)


def compute_levenshtein_distance(pred_seq, target_seq):
    """
    Computes the Levenshtein distance between two sequences of gesture IDs.

    Args:
        pred_seq (list[int]): List of predicted gesture IDs.
        target_seq (list[int]): List of ground truth gesture IDs.

    Returns:
        int: The edit distance.
    """
    # Filter out 0s (background) if they accidentally remain, though decoding usually handles this.
    # The metric is defined on the sequence of gestures (1-20).
    p_clean = [x for x in pred_seq if x != 0]
    t_clean = [x for x in target_seq if x != 0]

    return nltk.edit_distance(p_clean, t_clean)


def compute_dataset_score(predictions, targets):
    """
    Computes the overall error rate metric for the dataset.
    Metric = Sum(Levenshtein Distances) / Total Number of Ground Truth Gestures.

    Args:
        predictions (list[list[int]]): List of predicted gesture sequences.
        targets (list[list[int]]): List of ground truth gesture sequences.

    Returns:
        float: The computed score.
    """
    total_distance = 0
    total_gestures = 0

    for pred, target in zip(predictions, targets):
        # Ensure target is a list of ints
        if isinstance(target, (np.ndarray, torch.Tensor)):
            target = target.tolist()
        if isinstance(pred, (np.ndarray, torch.Tensor)):
            pred = pred.tolist()

        dist = compute_levenshtein_distance(pred, target)
        total_distance += dist

        # Count valid gestures in target (excluding background 0 if present,
        # though targets usually only contain 1-20 based on metadata)
        t_clean = [x for x in target if x != 0]
        total_gestures += len(t_clean)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures


def pad_sequence_batch(sequences, padding_value=0):
    """
    Pads a list of variable-length tensors to the maximum length in the batch.

    Args:
        sequences (list[torch.Tensor]): List of tensors (T, D) or (T,).
        padding_value (float): Value to pad with.

    Returns:
        tuple: (padded_tensor, lengths)
            padded_tensor: (Batch, MaxLen, ...)
            lengths: (Batch,)
    """
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_seqs = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=padding_value
    )
    return padded_seqs, lengths
