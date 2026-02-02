import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Computes the Mean F1-Score (samples average).

    Args:
        y_true: Ground truth binary labels (numpy array or torch tensor).
        y_pred_probs: Predicted probabilities (numpy array or torch tensor).
        threshold: Threshold to convert probabilities to binary predictions.

    Returns:
        float: The Mean F1-Score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()

    # Apply threshold
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    # Calculate F1 score with 'samples' average
    score = f1_score(y_true, y_pred_binary, average="samples", zero_division=0)
    return score


def save_checkpoint(model, optimizer, epoch, val_f1, path=Config.MODEL_SAVE_PATH):
    """
    Saves the model checkpoint including optimizer state and metrics.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_f1": val_f1,
    }
    torch.save(state, path)


def load_checkpoint(
    model, optimizer, path=Config.MODEL_SAVE_PATH, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Returns:
        tuple: (start_epoch, best_val_f1)
    """
    if not os.path.exists(path):
        return 0, 0.0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    val_f1 = checkpoint.get("val_f1", 0.0)

    return epoch, val_f1


def save_numpy_cache(data, path):
    """
    Saves data to a .npy file, creating directories if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)


def load_numpy_cache(path):
    """
    Loads data from a .npy file if it exists.

    Returns:
        numpy.ndarray or None: The loaded data, or None if file doesn't exist.
    """
    if os.path.exists(path):
        return np.load(path)
    return None


def create_submission(
    ids, pred_probs, mlb, threshold=0.5, output_path=Config.SUBMISSION_PATH
):
    """
    Generates the submission CSV file from predictions.

    Args:
        ids: Array of question Ids.
        pred_probs: Array of predicted probabilities.
        mlb: Fitted MultiLabelBinarizer instance.
        threshold: Probability threshold.
        output_path: Path to save the CSV.
    """
    # Convert probabilities to binary format
    pred_binary = (pred_probs >= threshold).astype(int)

    # Inverse transform to get list of tags
    # mlb.inverse_transform returns a list of tuples of tags
    pred_tags_tuples = mlb.inverse_transform(pred_binary)

    # Join tags with space
    pred_tags_strings = [" ".join(tags) for tags in pred_tags_tuples]

    # Create DataFrame
    df_sub = pd.DataFrame({"Id": ids, "Tags": pred_tags_strings})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_sub.to_csv(output_path, index=False)
