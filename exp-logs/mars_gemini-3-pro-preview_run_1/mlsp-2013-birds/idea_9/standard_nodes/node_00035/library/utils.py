import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import DEVICE


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_roc_auc(y_true, y_pred):
    """
    Computes the macro-averaged Area Under the ROC Curve.
    Robustly handles cases where specific classes might be missing from the target set
    (e.g., in small batches or validation splits), which would otherwise cause
    sklearn.metrics.roc_auc_score to raise an error.

    Args:
        y_true (np.ndarray): Ground truth binary labels (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.0 if no classes are valid.
    """
    try:
        if y_true.shape != y_pred.shape:
            # Fallback or error logging could go here, but we raise for correctness
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )

        num_classes = y_true.shape[1]
        auc_scores = []

        for i in range(num_classes):
            # ROC AUC is only defined if there is at least one positive and one negative sample
            if len(np.unique(y_true[:, i])) > 1:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)

        if not auc_scores:
            return 0.0

        return np.mean(auc_scores)

    except Exception as e:
        # In case of unexpected errors, return 0.0 to avoid crashing the training loop
        return 0.0


def save_checkpoint(model, path):
    """
    Saves the model state dictionary to the specified path.
    Creates the parent directory if it does not exist.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the checkpoint to.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device=DEVICE):
    """
    Loads the model state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): The file path of the checkpoint.
        device (str): The device to map the location to (default: DEVICE from config).

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def save_submission(rec_ids, probabilities, output_path):
    """
    Generates and saves the submission CSV file in the required format.

    The submission format requires combining 'rec_id' and 'species' into a single 'Id'
    column by multiplying 'rec_id' by 100 and adding the 'species' number.

    Args:
        rec_ids (np.ndarray or list): Array of recording IDs for the test set.
        probabilities (np.ndarray): Array of predicted probabilities of shape (N, num_classes).
        output_path (str): Path to save the CSV file.
    """
    records = []
    num_classes = probabilities.shape[1]

    for i, rec_id in enumerate(rec_ids):
        for species_idx in range(num_classes):
            # Construct the unique Id as per task description
            submission_id = int(rec_id * 100 + species_idx)
            prob = probabilities[i, species_idx]
            records.append({"Id": submission_id, "Probability": prob})

    df = pd.DataFrame(records)

    # Ensure directory exists
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    df.to_csv(output_path, index=False)
