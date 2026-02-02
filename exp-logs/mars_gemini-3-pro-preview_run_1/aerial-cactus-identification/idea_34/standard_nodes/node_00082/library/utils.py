import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Performs Mixup data augmentation.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for mixed inputs.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates Area Under the ROC Curve.
    y_true: ground truth labels (numpy array)
    y_scores: predicted probabilities (numpy array)
    """
    try:
        # Ensure inputs are numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_scores, torch.Tensor):
            y_scores = y_scores.detach().cpu().numpy()

        return roc_auc_score(y_true, y_scores)
    except ValueError:
        # Handle cases with only one class present in the batch
        return 0.5


def update_swa_bn(loader, model, device):
    """
    Updates Batch Normalization statistics for the SWA model.
    Iterates through the loader to compute running mean and variance.
    """
    model.train()  # Set to train mode to update BN stats

    # Reset BN running stats if possible (optional, but ensures clean stats)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            module.momentum = None  # Use cumulative moving average
            module.num_batches_tracked = torch.tensor(
                0, dtype=torch.long, device=device
            )

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            model(images)

    model.eval()


def save_checkpoint(model, path):
    """
    Saves the model state dictionary to the specified path.
    """
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    """
    Loads the model state dictionary from the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    state_dict = torch.load(path, map_location=device)

    # Sanitize state_dict for SWA/DataParallel checkpoints
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model
