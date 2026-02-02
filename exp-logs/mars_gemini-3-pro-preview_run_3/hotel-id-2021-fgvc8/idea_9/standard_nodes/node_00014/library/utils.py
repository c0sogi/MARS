import os
import random
import shutil
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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


def mean_average_precision(predictions, targets, k=5):
    """
    Calculates Mean Average Precision @ k (MAP@k).
    For this dataset, since there is only 1 ground truth label per image,
    this is equivalent to the Mean Reciprocal Rank (MRR) @ k.

    Args:
        predictions (torch.Tensor or np.ndarray): Shape (N, k) or (N, >k).
            Contains predicted class indices sorted by confidence (descending).
        targets (torch.Tensor or np.ndarray): Shape (N,).
            Contains ground truth class indices.
        k (int): Top k predictions to consider.

    Returns:
        float: The MAP@k score.
    """
    # Convert tensors to numpy arrays for easier processing
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure we are looking at the top k
    if predictions.shape[1] > k:
        predictions = predictions[:, :k]

    # If predictions have fewer than k columns, we just use what is available
    # (though typically the model should output at least k)

    score = 0.0
    num_samples = len(targets)

    for i in range(num_samples):
        target = targets[i]
        preds = predictions[i]

        # Find indices where the prediction matches the target
        # np.where returns a tuple of arrays
        matches = np.where(preds == target)[0]

        if len(matches) > 0:
            # The rank is the index (0-based) + 1
            # We take the first match (highest confidence)
            rank = matches[0] + 1
            score += 1.0 / rank

    return score / num_samples


def save_checkpoint(state, is_best, filepath, best_filepath=None):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model, optimizer, epoch, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): Path to save the current checkpoint.
        best_filepath (str, optional): Path to save the best checkpoint.
            If None, defaults to 'best_model.pth' in the same directory as filepath.
    """
    torch.save(state, filepath)
    if is_best:
        if best_filepath is None:
            dirname = os.path.dirname(filepath)
            best_filepath = os.path.join(dirname, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.
    Handles 'module.' prefix removal if model was saved with DataParallel but loaded without.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (object, optional): Scheduler to load state into.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Extract state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle 'module.' prefix mismatch
    # Case: Checkpoint has 'module.' (DataParallel), Model does not
    if list(state_dict.keys())[0].startswith("module.") and not hasattr(
        model, "module"
    ):
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        state_dict = new_state_dict

    # Load model weights
    model.load_state_dict(state_dict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
