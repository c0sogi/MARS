import os
import shutil
import torch
from library.config import set_seed, WORKING_DIR


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the specified directory.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, epoch, etc.
        is_best (bool): Boolean flag indicating if this is the best model so far.
        checkpoint_dir (str): Directory path where the checkpoint will be saved.
        filename (str): Name of the checkpoint file. Default is 'checkpoint.pth'.
    """
    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define file paths
    filepath = os.path.join(checkpoint_dir, filename)
    best_path = os.path.join(checkpoint_dir, "model_best.pth")

    # Save the state dictionary
    torch.save(state, filepath)

    # If this is the best model, create a copy
    if is_best:
        shutil.copyfile(filepath, best_path)


def load_checkpoint(filepath, model, optimizer=None, device=None):
    """
    Loads a model checkpoint from the specified file.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load state into.
        device (torch.device, optional): The device to map the checkpoint data to.
                                        Defaults to config.DEVICE if None.

    Returns:
        tuple: (start_epoch, best_score) extracted from the checkpoint.
               Returns (0, float('inf')) if keys are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback if the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Extract metadata
    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", float("inf"))

    return start_epoch, best_score


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
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
