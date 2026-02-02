import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import path_config, label_config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


class LabelManager:
    """
    Manages fine-grained labels (folder names) and their mapping to
    the 12 specific submission categories.
    """

    def __init__(self, load_cached_data=True):
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Ensure working directory exists for cache
        os.makedirs(path_config.working_dir, exist_ok=True)

        self._initialize_labels(load_cached_data)

    def _initialize_labels(self, load_cached_data):
        """
        Loads labels from cache or scans the directory structure.
        """
        cache_path = os.path.join(path_config.working_dir, "classes.parquet")

        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                self.classes = df["classes"].tolist()
            except Exception:
                # If cache is corrupted, re-scan
                self._scan_and_save(cache_path)
        else:
            self._scan_and_save(cache_path)

        # Build mappings
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(self.classes)}

    def _scan_and_save(self, cache_path):
        """
        Scans the input directory for folder names to determine the class vocabulary.
        """
        if not os.path.exists(path_config.train_audio_dir):
            # Fallback/Safety check
            raise FileNotFoundError(
                f"Training directory not found: {path_config.train_audio_dir}"
            )

        # List all subdirectories in the training audio folder
        subdirs = [
            d
            for d in os.listdir(path_config.train_audio_dir)
            if os.path.isdir(os.path.join(path_config.train_audio_dir, d))
        ]

        fine_grained_labels = set()

        for d in subdirs:
            # Skip background noise folder, it is used to synthesize 'silence'
            if d == "_background_noise_":
                continue
            fine_grained_labels.add(d)

        # Explicitly add the silence label
        fine_grained_labels.add(label_config.silence_label)

        # Sort for deterministic indexing
        self.classes = sorted(list(fine_grained_labels))

        # Save to parquet cache
        pd.DataFrame({"classes": self.classes}).to_parquet(cache_path)

    def get_num_classes(self):
        """Returns the total number of fine-grained classes."""
        return len(self.classes)

    def convert_label_to_idx(self, label):
        """Converts a string label to its integer index."""
        if label not in self.class_to_idx:
            raise ValueError(f"Label '{label}' not found in label set.")
        return self.class_to_idx[label]

    def convert_idx_to_label(self, idx):
        """Converts an integer index back to its string label."""
        if idx not in self.idx_to_class:
            raise ValueError(f"Index '{idx}' not found in label set.")
        return self.idx_to_class[idx]

    def map_to_submission_label(self, fine_grained_label):
        """
        Maps a fine-grained label (e.g., 'bird', 'up') to the 12 submission classes.

        Logic:
        - If label is a target command -> return label
        - If label is silence -> return silence
        - All other labels (bed, bird, etc.) -> return 'unknown'
        """
        # Check if it is one of the 10 target commands
        if fine_grained_label in label_config.target_labels:
            return fine_grained_label

        # Check if it is silence
        if fine_grained_label == label_config.silence_label:
            return label_config.silence_label

        # Otherwise, it is an auxiliary class that maps to unknown
        return label_config.unknown_label
