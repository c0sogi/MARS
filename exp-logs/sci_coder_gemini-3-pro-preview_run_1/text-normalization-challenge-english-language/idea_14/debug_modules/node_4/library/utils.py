import os
import random
import numpy as np
import torch
import re
from library.config import Config

# Pre-compile regex patterns defined in Config for efficiency
# This ensures we don't re-compile patterns for every token
COMPILED_REGEX_PATTERNS = [re.compile(pattern) for pattern in Config.REGEX_PATTERNS]


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_regex_features(token):
    """
    Generates a binary feature vector based on regex pattern matches for a given token.
    These features provide explicit morphological cues (e.g., is_digit, is_currency)
    to the model.

    Args:
        token (str): The input text token.

    Returns:
        np.ndarray: A 1D float32 array of shape (NUM_REGEX_FEATURES,) containing
                    1.0 for a match and 0.0 otherwise.
    """
    # Handle non-string inputs gracefully (though inputs should be strings)
    if not isinstance(token, str):
        token = str(token)

    features = []
    for pattern in COMPILED_REGEX_PATTERNS:
        # Use search to find the pattern anywhere, though many patterns have anchors
        if pattern.search(token):
            features.append(1.0)
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def compute_class_weights(class_counts):
    """
    Computes square-root smoothed class weights to handle class imbalance.
    Formula: Weight_c = sqrt(Total_Samples / Count_c)

    Args:
        class_counts (dict or pd.Series): A mapping of class labels to their frequency counts.

    Returns:
        dict: A dictionary mapping class labels to their computed weights.
    """
    # Convert to dictionary if it's a pandas Series
    if hasattr(class_counts, "to_dict"):
        counts = class_counts.to_dict()
    else:
        counts = class_counts

    total_samples = sum(counts.values())
    weights = {}

    for cls, count in counts.items():
        if count > 0:
            # Square-root smoothing dampens the effect of very rare classes
            # compared to simple inverse frequency
            weights[cls] = np.sqrt(total_samples / count)
        else:
            # Fallback for classes with 0 count (should not happen in training set)
            weights[cls] = 1.0

    return weights
