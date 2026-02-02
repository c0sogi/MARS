import os
import torch
import pandas as pd
import numpy as np
from library.config import set_seed, ID2LABEL, get_competition_label

# Expose set_seed for external use
__all__ = [
    "set_seed",
    "AverageMeter",
    "calculate_accuracy",
    "map_prediction_to_competition_label",
    "save_submission",
]


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy over an epoch.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Define Competition Labels
COMP_LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "silence",
    "unknown",
]
COMP_LABEL2ID = {k: v for v, k in enumerate(COMP_LABELS)}

# Precompute mapping from Fine-Grained ID (0-30) to Competition ID (0-11)
_map_list = []
for i in range(len(ID2LABEL)):
    fine_label = ID2LABEL[i]
    comp_label = get_competition_label(fine_label)
    _map_list.append(COMP_LABEL2ID[comp_label])

MAP_TENSOR = torch.tensor(_map_list, dtype=torch.long)


def calculate_accuracy(output, target):
    """
    Computes the accuracy based on the 12 competition labels.
    Maps 31-class logits/targets to 12-class space before comparison.

    Args:
        output (torch.Tensor): Model output logits [Batch, 31]
        target (torch.Tensor): Fine-grained target indices [Batch] (0-30)

    Returns:
        float: Accuracy value (0.0 to 1.0)
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred_fine = output.max(dim=1)

        # Ensure map_tensor is on correct device
        map_tensor = MAP_TENSOR.to(output.device)

        pred_comp = map_tensor[pred_fine]
        target_comp = map_tensor[target]

        correct = pred_comp.eq(target_comp).sum().item()
        return correct / batch_size


def map_prediction_to_competition_label(pred_idx):
    """
    Maps a predicted class index (from the 31 fine-grained classes)
    to the final 12-class competition label string.

    Args:
        pred_idx (int or torch.Tensor): The predicted index.

    Returns:
        str: The competition label (e.g., 'yes', 'no', 'unknown', 'silence').
    """
    if isinstance(pred_idx, torch.Tensor):
        pred_idx = pred_idx.item()

    # Get the fine-grained word (e.g., 'bed', 'yes', 'silence')
    fine_label = ID2LABEL.get(pred_idx, "unknown")

    # Map to competition format (e.g., 'bed' -> 'unknown', 'yes' -> 'yes')
    return get_competition_label(fine_label)


def save_submission(predictions, filenames, save_path):
    """
    Converts model predictions to the submission format and saves to CSV.

    Args:
        predictions (list or np.ndarray): List of predicted indices (0-30).
        filenames (list): List of corresponding filenames.
        save_path (str): Destination path for the CSV file.
    """
    mapped_labels = []

    for idx in predictions:
        label = map_prediction_to_competition_label(idx)
        mapped_labels.append(label)

    df = pd.DataFrame({"fname": filenames, "label": mapped_labels})

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)
