import os
import random
import numpy as np
import torch
from torch.optim.swa_utils import update_bn as torch_update_bn


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CuDNN.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, path):
    """
    Saves the training state (model, optimizer, scheduler, epoch, score) to a file.
    """
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    """
    Loads a checkpoint from the specified path into the model and optionally
    the optimizer and scheduler.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (torch.device, optional): Device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if requested and available
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if requested and available
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def average_weights(checkpoint_paths):
    """
    Computes the arithmetic mean of model weights from a list of checkpoint paths.
    Used for Stochastic Weight Averaging (SWA).

    Args:
        checkpoint_paths (list): List of file paths to the checkpoints.

    Returns:
        dict: A state_dict containing the averaged weights.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for averaging.")

    # Load the first checkpoint to serve as the base
    # We use CPU to avoid GPU memory overhead during the aggregation process
    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")
    avg_state_dict = first_ckpt["model_state_dict"]

    # Identify keys corresponding to floating point tensors (parameters/buffers)
    # We only average these; integer buffers (like num_batches_tracked) are left as is
    keys_to_avg = [
        k
        for k, v in avg_state_dict.items()
        if torch.is_tensor(v) and v.is_floating_point()
    ]

    # Initialize sums with the first model's weights (cloned to ensure we don't mutate the loaded dict)
    sums = {k: avg_state_dict[k].clone() for k in keys_to_avg}

    num_ckpts = len(checkpoint_paths)

    # Accumulate weights from remaining checkpoints
    if num_ckpts > 1:
        for path in checkpoint_paths[1:]:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt["model_state_dict"]
            for k in keys_to_avg:
                sums[k] += state[k]

        # Calculate the mean
        for k in keys_to_avg:
            avg_state_dict[k] = sums[k] / num_ckpts

    return avg_state_dict


def update_bn(loader, model, device):
    """
    Updates Batch Normalization statistics (running_mean, running_var) by performing
    a forward pass on the data loader. This is necessary after weight averaging.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing training data.
        model (torch.nn.Module): The model with averaged weights.
        device (torch.device): The device to run the update on.
    """
    # Uses PyTorch's built-in SWA utility to update BN statistics
    torch_update_bn(loader, model, device=device)
