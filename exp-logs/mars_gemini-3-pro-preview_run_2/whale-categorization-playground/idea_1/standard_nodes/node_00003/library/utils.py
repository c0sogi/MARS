import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_map5(targets, predictions):
    """
    Computes the Mean Average Precision @ 5 (MAP@5) score.

    For this specific task, there is only one ground truth label per image.
    The score for a single image is 1/(rank + 1) if the target is in the top 5,
    where rank is the 0-based index of the target in the predictions.
    Otherwise, the score is 0.

    Args:
        targets (list or np.ndarray): Ground truth labels.
        predictions (list of lists or np.ndarray): Top 5 predicted labels for each sample.

    Returns:
        float: The average MAP@5 score across all samples.
    """
    if len(targets) == 0:
        return 0.0

    score_sum = 0.0

    for target, preds in zip(targets, predictions):
        # Ensure we only look at the top 5
        top_preds = list(preds)[:5]

        if target in top_preds:
            rank = top_preds.index(target)
            score_sum += 1.0 / (rank + 1)

    return score_sum / len(targets)


def save_submission(image_names, predictions, filename=Config.SUBMISSION_PATH):
    """
    Formats and saves the submission file in the required CSV format.

    Args:
        image_names (list): List of image filenames (e.g., '00029b3a.jpg').
        predictions (list of lists): List of lists, where each inner list contains
                                     the top 5 predicted whale Ids (strings).
        filename (str): The path where the CSV will be saved.
    """
    # Format predictions: join the top 5 labels with spaces
    formatted_preds = [" ".join(p[:5]) for p in predictions]

    submission_df = pd.DataFrame({"Image": image_names, "Id": formatted_preds})

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    submission_df.to_csv(filename, index=False)


def predict_knn(
    test_embeddings,
    train_embeddings,
    train_labels,
    test_filenames=None,
    threshold=Config.NEW_WHALE_THRESHOLD,
    k=50,
):
    """
    Performs K-Nearest Neighbors search and applies thresholding logic for predictions.

    Args:
        test_embeddings (np.ndarray): Query embeddings.
        train_embeddings (np.ndarray): Reference embeddings.
        train_labels (np.ndarray): Reference labels.
        test_filenames (np.ndarray, optional): Filenames for test images.
        threshold (float): Distance threshold for 'new_whale'.
        k (int): Number of neighbors to retrieve.

    Returns:
        list: List of lists containing top 5 predicted IDs.
    """
    # Euclidean distance on L2-normalized embeddings is equivalent to ranking by Cosine Similarity
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(train_embeddings)

    distances, indices = knn.kneighbors(test_embeddings)

    final_predictions = []

    num_samples = len(test_embeddings)

    for i in range(num_samples):
        dists = distances[i]
        inds = indices[i]

        # Get the labels of the neighbors
        neighbor_labels = train_labels[inds]

        # Identify unique labels while preserving order (nearest first)
        unique_labels = []
        seen = set()
        for label in neighbor_labels:
            if label not in seen:
                unique_labels.append(label)
                seen.add(label)

        preds = []

        # Threshold Logic:
        # If the nearest neighbor is further away than the threshold,
        # assume it's a new whale not in the database.
        if dists[0] > threshold:
            preds.append("new_whale")

        # Fill the rest with the nearest unique neighbors
        for label in unique_labels:
            if label not in preds:
                preds.append(label)
                if len(preds) >= 5:
                    break

        # Fallback: If we still have fewer than 5 predictions,
        # append 'new_whale' if it's not already there.
        if len(preds) < 5 and "new_whale" not in preds:
            preds.append("new_whale")

        # Ensure we return exactly top 5
        final_predictions.append(preds[:5])

    return final_predictions


def load_metadata(split="train"):
    """
    Loads the metadata DataFrame for the specified split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

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
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)
