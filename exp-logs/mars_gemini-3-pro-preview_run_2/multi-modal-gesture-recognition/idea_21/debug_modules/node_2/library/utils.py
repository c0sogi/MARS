import os
import random
import numpy as np
import torch
import pandas as pd
import nltk
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein error rate (LER).

    The score is calculated as the sum of Levenshtein distances for all sequences
    divided by the total number of gestures in the ground truth.

    Args:
        predictions (list of list of int): List of predicted gesture label sequences.
        targets (list of list of int): List of ground truth gesture label sequences.

    Returns:
        float: The computed Levenshtein error rate.
    """
    total_distance = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        # Ensure inputs are lists and handle None
        p = list(pred) if pred is not None else []
        t = list(target) if target is not None else []

        # Compute edit distance between the two sequences of integers
        dist = nltk.edit_distance(p, t)

        total_distance += dist
        total_length += len(t)

    # Avoid division by zero
    if total_length == 0:
        return 0.0

    return total_distance / total_length


def format_submission(sample_ids, predictions, output_path):
    """
    Formats the predictions into a DataFrame and saves it to a CSV file.

    The output format follows the structure of 'randomPredictions.csv':
    - Column 'Id': Integer ID extracted from the sample ID (e.g., 'Sample00300' -> 300).
    - Column 'Sequence': Space-separated string of predicted gesture labels.

    Args:
        sample_ids (list of str): List of sample identifiers (e.g., 'Sample00300').
        predictions (list of list of int): List of predicted gesture sequences.
        output_path (str): The file path where the submission CSV will be saved.
    """
    formatted_ids = []
    formatted_seqs = []

    for sid, pred in zip(sample_ids, predictions):
        # Extract numeric ID from 'SampleXXXXX' format
        if isinstance(sid, str) and sid.startswith("Sample"):
            try:
                numeric_id = int(sid.replace("Sample", ""))
            except ValueError:
                # Fallback if format is unexpected
                numeric_id = sid
        else:
            numeric_id = sid

        formatted_ids.append(numeric_id)

        # Convert prediction list to space-separated string
        if pred:
            seq_str = " ".join(map(str, pred))
        else:
            seq_str = ""
        formatted_seqs.append(seq_str)

    # Create DataFrame
    df = pd.DataFrame({"Id": formatted_ids, "Sequence": formatted_seqs})

    # Sort by Id to ensure consistent order
    # Ensure Id is treated as int for sorting if possible
    try:
        df["Id"] = df["Id"].astype(int)
        df = df.sort_values("Id")
    except ValueError:
        pass  # Keep original order or string sort if conversion fails

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
