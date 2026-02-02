import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
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
    Computes and stores the average and current value of a metric.
    Useful for tracking losses and metrics during training.
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


def weight_soup(checkpoint_paths):
    """
    Averages the weights of multiple PyTorch model checkpoints (Model Soup).

    Args:
        checkpoint_paths (list of str): List of file paths to the checkpoint .pth files.

    Returns:
        dict: A state dictionary containing the averaged weights, ready to be loaded into a model.
    """
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths list cannot be empty.")

    # Load the first checkpoint
    # Map to CPU to avoid GPU OOM issues during the averaging process
    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")

    # Extract state_dict if the checkpoint is a dictionary containing metadata
    if "state_dict" in first_ckpt:
        avg_state_dict = first_ckpt["state_dict"]
    elif "model_state_dict" in first_ckpt:
        avg_state_dict = first_ckpt["model_state_dict"]
    elif "model" in first_ckpt:
        avg_state_dict = first_ckpt["model"]
    else:
        avg_state_dict = first_ckpt

    # Clone and convert to float64 for higher precision accumulation
    # We filter to ensure we only process tensors
    avg_state_dict = {
        k: v.clone().to(torch.float64)
        for k, v in avg_state_dict.items()
        if isinstance(v, torch.Tensor)
    }

    num_models = len(checkpoint_paths)

    # If only one model, return it (casted back to float32)
    if num_models == 1:
        return {k: v.to(torch.float32) for k, v in avg_state_dict.items()}

    # Iterate over the remaining checkpoints
    for i in range(1, num_models):
        ckpt = torch.load(checkpoint_paths[i], map_location="cpu")

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        for k in avg_state_dict:
            if k in state_dict:
                avg_state_dict[k] += state_dict[k]

    # Compute average and convert back to float32
    for k in avg_state_dict:
        avg_state_dict[k] = (avg_state_dict[k] / num_models).to(torch.float32)

    return avg_state_dict
