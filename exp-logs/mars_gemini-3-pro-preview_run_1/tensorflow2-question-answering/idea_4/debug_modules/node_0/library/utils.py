import os
import logging
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from library.config import Config, set_seed


def setup_logger(name="daan_logger", log_file=None, level=logging.INFO):
    """
    Sets up a logger to write to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, logs only to console.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        loss (float): Validation loss.
        filepath (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str or torch.device): Device to map the location to.

    Returns:
        int: The epoch associated with the checkpoint (or 0 if not found).
        float: The loss associated with the checkpoint (or None).
    """
    if not os.path.exists(filepath):
        return 0, None

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("loss", None)


def compute_classification_metrics(y_true, y_pred, threshold=0.5):
    """
    Computes metrics for binary classification (Long Answer).

    Args:
        y_true (np.array): Ground truth labels (0 or 1).
        y_pred (np.array): Predicted probabilities.
        threshold (float): Threshold for converting probabilities to classes.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1.
    """
    y_pred_bin = (y_pred >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def compute_span_overlap_f1(pred_span, true_spans):
    """
    Computes F1 score based on token overlap between a predicted span and a set of true spans.
    Used for Short Answer evaluation.

    Args:
        pred_span (tuple): (start_index, end_index) of prediction.
        true_spans (list of tuples): List of (start_index, end_index) ground truths.

    Returns:
        float: Best F1 score against any of the ground truths.
    """
    if not true_spans:
        # If no ground truth, F1 is 0 unless prediction is also empty (which is handled outside)
        return 0.0

    pred_tokens = set(range(pred_span[0], pred_span[1] + 1))

    best_f1 = 0.0
    for true_span in true_spans:
        true_tokens = set(range(true_span[0], true_span[1] + 1))

        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            f1 = 1.0 if len(pred_tokens) == len(true_tokens) else 0.0
        else:
            common = pred_tokens.intersection(true_tokens)
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(true_tokens)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1

    return best_f1


def save_data_cache(data, filepath):
    """
    Saves data to a cache file (parquet or npy).
    Ensures directory exists.

    Args:
        data (pd.DataFrame or np.ndarray): Data to save.
        filepath (str): Destination path.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if filepath.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError("Data must be a pandas DataFrame for .parquet extension.")
    elif filepath.endswith(".npy"):
        if isinstance(data, np.ndarray):
            np.save(filepath, data)
        else:
            raise ValueError("Data must be a numpy array for .npy extension.")
    else:
        raise ValueError("Unsupported file extension. Use .parquet or .npy")


def load_data_cache(filepath):
    """
    Loads data from a cache file.

    Args:
        filepath (str): Path to file.

    Returns:
        pd.DataFrame or np.ndarray: Loaded data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cache file not found: {filepath}")

    if filepath.endswith(".parquet"):
        return pd.read_parquet(filepath)
    elif filepath.endswith(".npy"):
        return np.load(
            filepath, allow_pickle=False
        )  # allow_pickle=False for security/strictness
    else:
        raise ValueError("Unsupported file extension. Use .parquet or .npy")


def format_submission_file(predictions, output_path):
    """
    Formats and saves the submission file.

    Args:
        predictions (dict): Dictionary mapping example_id to prediction strings.
                            Keys should be like "-7853356005143141653_long".
        output_path (str): Path to save the CSV.
    """
    # Convert dict to DataFrame
    df = pd.DataFrame(
        list(predictions.items()), columns=["example_id", "PredictionString"]
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)
