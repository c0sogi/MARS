import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0, device=None):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Target labels.
        alpha (float): Parameter for the Beta distribution.
        device (torch.device): Device to move the data to.

    Returns:
        mixed_x (torch.Tensor): Mixed input images.
        y_a (torch.Tensor): Labels for the first component.
        y_b (torch.Tensor): Labels for the second component.
        lam (float): The mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss.

    Args:
        criterion (callable): The loss function (e.g., CrossEntropyLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Labels for the first component.
        y_b (torch.Tensor): Labels for the second component.
        lam (float): The mixing coefficient lambda.

    Returns:
        loss (torch.Tensor): The weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def average_weights(checkpoint_paths):
    """
    Averages the weights of multiple model checkpoints for SWA.

    Args:
        checkpoint_paths (list of str): List of file paths to the model checkpoints.

    Returns:
        avg_state_dict (dict): The averaged state dictionary.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoints provided for averaging.")

    # Load the first model to initialize the average
    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")
    avg_state_dict = first_ckpt

    # Convert parameters to float for averaging
    for key in avg_state_dict:
        if isinstance(avg_state_dict[key], torch.Tensor):
            avg_state_dict[key] = avg_state_dict[key].float()

    # Accumulate remaining models
    for i in range(1, len(checkpoint_paths)):
        ckpt = torch.load(checkpoint_paths[i], map_location="cpu")
        for key in avg_state_dict:
            if isinstance(avg_state_dict[key], torch.Tensor):
                avg_state_dict[key] += ckpt[key].float()

    # Divide by number of models
    n_models = len(checkpoint_paths)
    for key in avg_state_dict:
        if isinstance(avg_state_dict[key], torch.Tensor):
            avg_state_dict[key] /= n_models

    return avg_state_dict


def update_bn(loader, model, device):
    """
    Updates the Batch Normalization statistics for the SWA model.
    This custom implementation handles the specific (image, metadata) input signature.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the training set.
        model (nn.Module): The model with averaged weights.
        device (torch.device): The device to run the computation on.
    """
    model.train()  # Set to train mode to update BN stats

    # Reset BN running statistics
    # This ensures we calculate fresh stats based on the averaged weights
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            module.momentum = None  # Use simple average
            module.reset_running_stats()

    with torch.no_grad():
        for batch in loader:
            # Unpack batch based on the expected format: (images, metadata, labels)
            # We only need inputs for the forward pass
            images = batch[0].to(device)
            metadata = batch[1].to(device)

            # Forward pass to update BN stats
            # The model handles the internal routing
            _ = model(images, metadata)
