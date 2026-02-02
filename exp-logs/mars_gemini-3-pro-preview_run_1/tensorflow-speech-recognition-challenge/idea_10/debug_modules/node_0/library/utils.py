import os
import random
import numpy as np
import torch
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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


class LabelMapper:
    """
    Handles the conversion between fine-grained training labels (31 classes)
    and the final submission labels (12 classes).
    """

    def __init__(self):
        self.all_labels = Config.ALL_LABELS
        self.target_labels = set(Config.TARGET_LABELS)
        self.silence_label = Config.SILENCE_LABEL
        self.unknown_label = Config.UNKNOWN_LABEL

        # Create mappings
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.all_labels)}

    def to_index(self, label):
        """
        Converts a fine-grained string label to its integer index.

        Args:
            label (str): The label string (e.g., 'yes', 'bed').

        Returns:
            int: The corresponding index.
        """
        if label not in self.label_to_idx:
            raise ValueError(f"Label '{label}' not found in configuration.")
        return self.label_to_idx[label]

    def to_label(self, idx):
        """
        Converts an integer index back to its fine-grained string label.

        Args:
            idx (int): The class index.

        Returns:
            str: The corresponding label string.
        """
        if idx not in self.idx_to_label:
            raise ValueError(f"Index '{idx}' is out of bounds.")
        return self.idx_to_label[idx]

    def map_to_submission(self, fine_grained_label):
        """
        Maps a fine-grained training label to the 12-class submission format.

        Logic:
        - Target labels ('yes', 'no', etc.) -> themselves
        - 'silence' -> 'silence'
        - Auxiliary labels ('bed', 'bird', etc.) -> 'unknown'

        Args:
            fine_grained_label (str): The label used during training.

        Returns:
            str: The label string for submission.
        """
        if fine_grained_label in self.target_labels:
            return fine_grained_label
        elif fine_grained_label == self.silence_label:
            return self.silence_label
        else:
            return self.unknown_label

    def index_to_submission(self, idx):
        """
        Directly maps a predicted index to the submission label string.

        Args:
            idx (int): The predicted class index.

        Returns:
            str: The label string for submission.
        """
        fine_label = self.to_label(idx)
        return self.map_to_submission(fine_label)
