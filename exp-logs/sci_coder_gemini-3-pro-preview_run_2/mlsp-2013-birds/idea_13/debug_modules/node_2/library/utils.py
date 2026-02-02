import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Seeds Python, NumPy, and PyTorch for reproducibility.

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


def worker_init_fn(worker_id):
    """
    Worker initialization function for PyTorch DataLoaders to ensure
    each worker has a different random seed based on the global seed.

    Args:
        worker_id (int): The ID of the worker.
    """
    # Get the base seed from torch's initial seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC) for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth labels (N, Num_Classes).
        y_pred (np.ndarray): Predicted probabilities (N, Num_Classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        # average='macro' calculates metrics for each label, and finds their unweighted mean.
        # This does not take label imbalance into account.
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # This can happen if a class is not present in the batch/set
        # In such cases, we try to compute it ignoring the missing class or return 0.5
        # For robustness in small batches, we handle it gracefully
        score = 0.5

    return score


def mixup_data(x, y, alpha=0.4, device="cpu"):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input batch of images/spectrograms.
        y (torch.Tensor): Input batch of labels.
        alpha (float): Mixup interpolation coefficient parameter.
        device (str or torch.device): Device to perform computations on.

    Returns:
        mixed_x (torch.Tensor): Mixed inputs.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Permuted labels.
        lam (float): Lambda value used for mixing.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def get_pos_weights(y_train, device="cpu"):
    """
    Calculates positive weights for BCEWithLogitsLoss based on class imbalance.
    Weight = (Number of Negatives) / (Number of Positives)

    Args:
        y_train (np.ndarray or pd.DataFrame): The training labels matrix (N_samples, N_classes).
        device (str or torch.device): Device to put the weights tensor on.

    Returns:
        torch.Tensor: A tensor of weights for each class.
    """
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)

    # Count positives for each class
    pos_counts = np.sum(y_train, axis=0)
    total_samples = y_train.shape[0]
    neg_counts = total_samples - pos_counts

    # Calculate weights: neg / pos
    # Add a small epsilon to avoid division by zero
    weights = neg_counts / (pos_counts + 1e-6)

    return torch.as_tensor(weights, dtype=torch.float32).to(device)
