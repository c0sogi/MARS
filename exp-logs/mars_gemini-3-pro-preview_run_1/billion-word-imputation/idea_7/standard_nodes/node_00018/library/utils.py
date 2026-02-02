import os
import random
import numpy as np
import torch
import nltk
from collections import defaultdict
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The computing device.
    """
    return torch.device(Config.DEVICE)


class MetricMonitor:
    """
    Tracks and averages metrics (e.g., loss, accuracy) during training/validation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all tracked metrics."""
        self.metrics = defaultdict(lambda: {"sum": 0.0, "count": 0, "avg": 0.0})

    def update(self, metrics):
        """
        Updates the running average for the provided metrics.

        Args:
            metrics (dict): A dictionary where keys are metric names and values are the current observations.
        """
        for name, value in metrics.items():
            metric_data = self.metrics[name]
            metric_data["sum"] += value
            metric_data["count"] += 1
            metric_data["avg"] = metric_data["sum"] / metric_data["count"]

    def get_metrics(self):
        """
        Returns a dictionary of the current average values for all metrics.

        Returns:
            dict: Mapping of metric name to average value.
        """
        return {k: v["avg"] for k, v in self.metrics.items()}

    def __str__(self):
        """
        Returns a string representation of the metrics with full precision.
        """
        return " | ".join([f"{k}: {v['avg']}" for k, v in self.metrics.items()])


def get_pos_tagger():
    """
    Initializes and returns a POS tagging function using NLTK.
    Ensures necessary NLTK resources (taggers, tokenizers) are available.

    Returns:
        function: A callable that takes a list of tokens (str) and returns a list of (word, tag) tuples.
    """
    # Required NLTK resources
    resources = [
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/universal_tagset", "universal_tagset"),
        ("tokenizers/punkt", "punkt"),
    ]

    # Ensure resources are downloaded
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception as e:
                print(
                    f"Warning: Failed to download NLTK resource '{resource_name}': {e}"
                )

    def tagger(tokens):
        """
        Tags a list of tokens with Universal POS tags.

        Args:
            tokens (list): List of string tokens.

        Returns:
            list: List of (word, tag) tuples using the Universal tagset.
        """
        # Use universal tagset to simplify tags (e.g., 'NOUN', 'VERB')
        return nltk.pos_tag(tokens, tagset="universal")

    return tagger
