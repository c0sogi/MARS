import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode ensures reproducibility but may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint. If this is the best model, creates a copy at the
    location specified by Config.MODEL_CHECKPOINT_PATH.

    Args:
        state (dict): State dictionary containing model parameters, optimizer, epoch, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file to save in the working directory.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    filepath = os.path.join(Config.WORKING_DIR, filename)

    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, Config.MODEL_CHECKPOINT_PATH)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Loads a model checkpoint from the specified path.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (object, optional): The learning rate scheduler to load state into.

    Returns:
        tuple: (start_epoch, best_f1)
            - start_epoch (int): The epoch to resume from.
            - best_f1 (float): The best F1 score recorded in the checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        print(f"[-] No checkpoint found at '{checkpoint_path}'")
        return 0, 0.0

    print(f"[+] Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_f1 = checkpoint.get("best_f1", 0.0)

    print(f"    Loaded checkpoint (epoch {start_epoch}, best_f1 {best_f1})")
    return start_epoch, best_f1


def calculate_metrics(y_true, y_pred):
    """
    Calculates the Macro F1 score for the given predictions.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth labels.
        y_pred (torch.Tensor or np.ndarray): Predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate Macro F1 score
    return f1_score(y_true, y_pred, average="macro")
