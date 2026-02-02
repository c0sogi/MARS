import os
import random
import bisect
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_inversions(a):
    """
    Counts the number of inversions in a list of integers.
    Uses a bisect approach which is effectively O(N^2) due to list insertion,
    but sufficiently fast for notebook cell counts (typically N < 200).

    Args:
        a (list[int]): A list of integers (ranks).

    Returns:
        int: The number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find the position where x should be inserted to maintain order
        idx = bisect.bisect_right(sorted_so_far, x)
        # Elements to the right of idx in sorted_so_far are greater than x
        # and were seen before x, so they form inversions.
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(predictions, ground_truths):
    """
    Computes the global Kendall Tau correlation metric as defined in the task.
    K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        predictions (list of list of str): Predicted cell orders (lists of cell IDs).
        ground_truths (list of list of str): Ground truth cell orders (lists of cell IDs).

    Returns:
        float: The Kendall Tau score.
    """
    total_swaps = 0
    total_denominator = 0

    for pred, true in zip(predictions, ground_truths):
        n = len(true)
        if n <= 1:
            continue

        # Create a mapping from cell_id to its correct rank (0 to n-1)
        rank_map = {cell_id: i for i, cell_id in enumerate(true)}

        # Convert prediction sequence to rank sequence
        # We filter to ensure robustness, though pred should be a permutation of true
        pred_ranks = [rank_map[cid] for cid in pred if cid in rank_map]

        # Count inversions (S)
        swaps = count_inversions(pred_ranks)

        # Denominator term for this notebook: n * (n - 1)
        n_n_minus_1 = n * (n - 1)

        total_swaps += swaps
        total_denominator += n_n_minus_1

    if total_denominator == 0:
        return 1.0

    score = 1 - 4 * (total_swaps / total_denominator)
    return score


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences of code and markdown cells.

    Args:
        batch (list of dict): List of samples, where each sample is a dict containing:
            - 'code_emb': Tensor of shape (num_code, dim)
            - 'md_emb': Tensor of shape (num_md, dim)
            - 'labels': Tensor of shape (num_md,) [Optional]

    Returns:
        dict: Batched data containing:
            - 'code_emb': Padded tensor (Batch, Max_Code, Dim)
            - 'code_mask': Boolean mask (Batch, Max_Code) - True for real tokens
            - 'md_emb': Padded tensor (Batch, Max_Md, Dim)
            - 'md_mask': Boolean mask (Batch, Max_Md) - True for real tokens
            - 'labels': Padded tensor (Batch, Max_Md) with -100 padding [Optional]
    """
    code_embs = [item["code_emb"] for item in batch]
    md_embs = [item["md_emb"] for item in batch]

    # Pad embeddings
    # batch_first=True results in (Batch, Max_Len, Dim)
    code_padded = pad_sequence(code_embs, batch_first=True, padding_value=0.0)
    md_padded = pad_sequence(md_embs, batch_first=True, padding_value=0.0)

    # Create masks (True where data exists, False where padded)
    batch_size = len(batch)

    max_code_len = code_padded.size(1)
    code_lens = torch.tensor([len(x) for x in code_embs])
    # Create mask: [0, 1, ..., max-1] < len
    code_mask = torch.arange(max_code_len).expand(
        batch_size, max_code_len
    ) < code_lens.unsqueeze(1)

    max_md_len = md_padded.size(1)
    md_lens = torch.tensor([len(x) for x in md_embs])
    md_mask = torch.arange(max_md_len).expand(
        batch_size, max_md_len
    ) < md_lens.unsqueeze(1)

    result = {
        "code_emb": code_padded,
        "code_mask": code_mask,
        "md_emb": md_padded,
        "md_mask": md_mask,
    }

    # Handle labels if present
    if "labels" in batch[0]:
        labels = [item["labels"] for item in batch]
        # Pad labels with -100 (standard ignore_index for CrossEntropyLoss)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        result["labels"] = labels_padded

    return result
