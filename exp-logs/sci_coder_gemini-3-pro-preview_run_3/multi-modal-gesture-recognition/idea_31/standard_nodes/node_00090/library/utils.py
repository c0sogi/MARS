import os
import random
import json
import numpy as np
import pandas as pd
import torch
import nltk
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_normalized_levenshtein(predictions, ground_truths):
    """
    Computes the metric: Total Levenshtein Distance / Total Ground Truth Gestures.

    Args:
        predictions (list of list of int): Predicted gesture sequences.
        ground_truths (list of list of int): Ground truth gesture sequences.

    Returns:
        float: The normalized error rate.
    """
    total_distance = 0
    total_gestures = 0

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Mismatch in number of samples: preds={len(predictions)}, gt={len(ground_truths)}"
        )

    for pred, gt in zip(predictions, ground_truths):
        # Calculate edit distance using NLTK
        dist = nltk.edit_distance(pred, gt)
        total_distance += dist
        total_gestures += len(gt)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures


def compute_class_weights(
    metadata_path=Config.TRAIN_METADATA_PATH, load_cached_data=True
):
    """
    Computes inverse frequency weights for the classes based on frame counts in the training data.
    Returns a torch tensor of shape (NUM_CLASSES,).

    The background class (0) weight is taken from Config.LOSS_BG_WEIGHT.
    Classes 1-20 are weighted inversely proportional to their total duration (frames).

    Implements caching to avoid re-parsing metadata.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "class_weights.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            return torch.tensor(weights_np, dtype=torch.float32)
        except Exception:
            pass  # Fallback to compute if load fails

    # 2. Compute from scratch
    if not os.path.exists(metadata_path):
        # Fallback if metadata missing (e.g. unit testing)
        return torch.ones(Config.NUM_CLASSES)

    df = pd.read_csv(metadata_path)

    # Initialize counts for classes 1 to 20
    # We use a dictionary to accumulate frame durations
    class_frame_counts = {i: 0 for i in range(1, Config.NUM_CLASSES)}

    for _, row in df.iterrows():
        try:
            labels_str = row.get("labels", "[]")
            if not isinstance(labels_str, str):
                continue
            labels = json.loads(labels_str)
            for label in labels:
                lid = label.get("id")
                start = label.get("begin")
                end = label.get("end")

                if lid is not None and start is not None and end is not None:
                    duration = end - start + 1
                    if lid in class_frame_counts:
                        class_frame_counts[lid] += duration
        except Exception:
            continue

    # Convert to array for classes 1-20
    counts = np.array([class_frame_counts[i] for i in range(1, Config.NUM_CLASSES)])

    # Handle zero counts (unlikely but possible in subsets) by clamping to 1
    counts = np.maximum(counts, 1)

    # Compute inverse weights normalized by median to keep scale around 1.0
    median_count = np.median(counts)
    weights_1_to_20 = median_count / counts

    # Construct full weight vector
    full_weights = np.zeros(Config.NUM_CLASSES)
    full_weights[0] = Config.LOSS_BG_WEIGHT
    full_weights[1:] = weights_1_to_20

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, full_weights)

    return torch.tensor(full_weights, dtype=torch.float32)
