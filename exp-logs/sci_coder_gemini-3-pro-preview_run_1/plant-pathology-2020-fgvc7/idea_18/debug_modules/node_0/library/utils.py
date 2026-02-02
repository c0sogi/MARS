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

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, C).

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check if we have valid data
    if len(y_true) == 0:
        return 0.0

    try:
        # average='macro' computes the metric for each label, and finds their unweighted mean.
        # This matches "Mean column-wise ROC AUC".
        score = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
    except ValueError:
        # Handle cases where a class might not be present in the batch
        score = 0.0

    return score


def get_class_weights(df, target_cols, load_cached_data=True):
    """
    Calculates class weights inversely proportional to class frequencies.
    Implements caching using .npy format.

    Args:
        df (pd.DataFrame): The training dataframe containing target columns.
        target_cols (list): List of column names corresponding to the targets.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the weights.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "class_weights.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            # print(f"Loaded class weights from cache: {cache_path}")
            return torch.tensor(weights_np, dtype=torch.float32)
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from scratch
    # Assuming One-Hot or similar encoding where we can sum the columns
    # If df contains the targets directly
    class_counts = df[target_cols].sum().values
    total_samples = len(df)
    num_classes = len(target_cols)

    # Formula: w_j = n_samples / (n_classes * n_samples_j)
    # Add a small epsilon to avoid division by zero if a class is missing (unlikely in this dataset)
    weights_np = total_samples / (num_classes * (class_counts + 1e-6))

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, weights_np)
    # print(f"Computed and saved class weights to: {cache_path}")

    return torch.tensor(weights_np, dtype=torch.float32)


def check_initial_loss(model, data_loader, criterion, device):
    """
    Runs a forward pass on a single batch to verify the initial loss is reasonable.
    Reasonable baseline for 4 classes is -ln(1/4) ~= 1.386.

    Args:
        model (torch.nn.Module): The model to check.
        data_loader (torch.utils.data.DataLoader): DataLoader to retrieve a batch from.
        criterion (callable): The loss function.
        device (torch.device): The device to run on.

    Returns:
        float: The initial loss value.
    """
    model.eval()

    try:
        # Get one batch
        images, labels = next(iter(data_loader))
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss_val = loss.item()

        # Random guessing baseline for 4 classes
        baseline = -np.log(1.0 / 4.0)

        print(
            f"Initial Loss Check: {loss_val:.6f} (Random Guessing Baseline: ~{baseline:.4f})"
        )

        if loss_val > (baseline * 1.5):
            print(
                "WARNING: Initial loss is significantly higher than random guessing. Check initialization or preprocessing."
            )
        elif loss_val < (baseline * 0.1):
            print("WARNING: Initial loss is suspiciously low. Check for data leakage.")
        else:
            print("Initial loss is within a reasonable range.")

        return loss_val

    except Exception as e:
        print(f"Failed to check initial loss: {e}")
        return float("inf")
    finally:
        # Ensure model is returned to training mode if intended,
        # though usually the training loop sets this explicitly.
        model.train()
