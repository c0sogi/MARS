import os
import pandas as pd
import numpy as np
import torch
from library.config import Config, set_seed


def get_metadata(split="train"):
    """
    Loads the metadata CSV file for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = Config.TRAIN_CSV
    elif split == "val":
        path = Config.VAL_CSV
    elif split == "test":
        path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


class FineGrainedLabelEncoder:
    """
    Encodes fine-grained labels (folder names) into integers for training
    and maps them to the final 12-class competition targets for submission.

    This supports the strategy of training on all 30+ available classes to
    learn better features, then collapsing non-target classes to 'unknown'.
    """

    def __init__(self):
        self.labels = []
        self.label2id = {}
        self.id2label = {}
        self.target_labels = Config.TARGET_LABELS

    def fit(self, metadata_df):
        """
        Scans the metadata dataframe to find all unique fine-grained labels
        based on the directory structure in 'filepath'.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing a 'filepath' column.
        """
        unique_labels = set()

        # Extract labels from file paths
        # Format expected: train/audio/<label>/<filename>
        for filepath in metadata_df["filepath"]:
            # Normalize path to handle OS differences
            norm_path = os.path.normpath(filepath)
            parts = norm_path.split(os.sep)

            # We expect at least: train/audio/label/file.wav
            if len(parts) >= 2:
                # The label is the parent folder of the file
                folder_name = parts[-2]

                # Handle special folders
                if folder_name == "_background_noise_":
                    unique_labels.add("silence")
                elif folder_name == "audio":
                    # Skip if parsing test files (test/audio/clip.wav)
                    continue
                else:
                    unique_labels.add(folder_name)

        # Sort for deterministic index assignment
        self.labels = sorted(list(unique_labels))

        # Build mappings
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}

    def transform(self, labels):
        """
        Converts a list of string labels to class IDs.

        Args:
            labels (list): List of label strings.

        Returns:
            list: List of integer class IDs.
        """
        return [self.label2id[label] for label in labels]

    def inverse_transform(self, ids):
        """
        Converts class IDs back to string labels.

        Args:
            ids (list, np.ndarray, or torch.Tensor): Class IDs.

        Returns:
            list: List of label strings.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()
        elif isinstance(ids, (int, np.integer)):
            ids = [ids]

        return [self.id2label[int(i)] for i in ids]

    def map_to_target(self, fine_label):
        """
        Maps a fine-grained label (e.g., 'bed', 'up', 'silence')
        to the 12-class competition format.

        Logic:
        - If label is in TARGET_LABELS ('yes', 'no', 'up', ...): keep it.
        - If label is 'silence': keep it.
        - All other labels ('bed', 'bird', etc.): map to 'unknown'.

        Args:
            fine_label (str): The fine-grained label.

        Returns:
            str: The competition target label.
        """
        if fine_label in self.target_labels:
            return fine_label
        elif fine_label == "silence":
            return "silence"
        else:
            return "unknown"

    def __len__(self):
        return len(self.labels)


def map_predictions_to_submission(predictions, label_encoder):
    """
    Helper function to convert model output IDs to submission strings.

    Args:
        predictions (list/array): Array of predicted class IDs (fine-grained).
        label_encoder (FineGrainedLabelEncoder): Fitted encoder instance.

    Returns:
        list: List of mapped strings ready for submission (e.g., 'yes', 'unknown').
    """
    # 1. Convert IDs to Fine-Grained Labels (e.g., 0 -> 'bed')
    fine_labels = label_encoder.inverse_transform(predictions)

    # 2. Map Fine-Grained Labels to Target Labels (e.g., 'bed' -> 'unknown')
    submission_labels = [label_encoder.map_to_target(l) for l in fine_labels]

    return submission_labels
