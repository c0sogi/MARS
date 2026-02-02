import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_class_weights(train_csv_path=None):
    """
    Calculates inverse frequency class weights based on the training metadata.

    Args:
        train_csv_path (str, optional): Path to the training metadata CSV.
                                        Defaults to Config.METADATA_DIR/train.csv.

    Returns:
        torch.Tensor: A tensor containing the weights for each class,
                      ordered according to Config.CLASSES.
    """
    if train_csv_path is None:
        train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training metadata not found at {train_csv_path}")

    df = pd.read_csv(train_csv_path)

    total_samples = len(df)
    class_counts = []

    # Ensure we iterate in the order defined by Config.CLASSES
    for class_name in Config.CLASSES:
        if class_name in df.columns:
            count = df[class_name].sum()
            class_counts.append(count)
        else:
            # Fallback if column not found (should not happen based on metadata)
            print(f"Warning: Class column '{class_name}' not found in metadata.")
            class_counts.append(1)  # Avoid division by zero

    class_counts = np.array(class_counts)

    # Inverse frequency: Total / Count
    # Add a small epsilon to count to prevent division by zero if a class is empty
    weights = total_samples / (class_counts + 1e-6)

    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the model and training state to a file.

    Args:
        state (dict): Dictionary containing model_state_dict, optimizer_state_dict, etc.
        filename (str): Name of the file to save.
    """
    # Ensure directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    filepath = os.path.join(Config.WORK_DIR, filename)
    torch.save(state, filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        checkpoint_path (str): Full path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The full loaded checkpoint dictionary (useful for retrieving epoch/score).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def compute_metric(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC.

    Args:
        y_true (np.ndarray): Ground truth labels of shape (N, Num_Classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, Num_Classes).

    Returns:
        float: The mean ROC AUC score.
    """
    # Check if y_true and y_pred have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate ROC AUC for each column and take the average (macro)
    # Handling potential edge cases where a class might only have one label in the batch
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback for batches with single-class targets (mostly relevant during small batch debugging)
        # We calculate per column and ignore columns with only one class present
        scores = []
        for i in range(y_true.shape[1]):
            try:
                if len(np.unique(y_true[:, i])) > 1:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
            except ValueError:
                pass

        if len(scores) > 0:
            score = np.mean(scores)
        else:
            score = 0.5  # Default random guess score if undefined

    return score
