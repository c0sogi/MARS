import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_bn(loader, model, device=None):
    """
    Updates the Batch Normalization statistics (running_mean and running_var)
    for the model by performing a forward pass on the data loader.
    This is essential for SWA (Stochastic Weight Averaging) models.

    Args:
        loader (DataLoader): The training data loader.
        model (nn.Module): The model with averaged weights.
        device (torch.device, optional): The device to run the model on.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    # Reset BN running stats and momentum
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    # Forward pass to accumulate stats
    with torch.no_grad():
        for batch in loader:
            # Handle (input, label) tuples or just input
            if isinstance(batch, (list, tuple)):
                input_data = batch[0]
            else:
                input_data = batch

            input_data = input_data.to(device)
            model(input_data)

    # Restore original momentum
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]


def map_to_competition_label(label_str):
    """
    Maps a fine-grained class label (e.g., 'bed', 'yes') to the
    competition's 12-class format.

    Logic:
    - If label is in TARGET_LABELS ('yes', 'no', 'up', ...), keep it.
    - If label is 'silence', keep it.
    - Otherwise (e.g., 'bed', 'bird'), map to 'unknown'.

    Args:
        label_str (str): The predicted label string.

    Returns:
        str: The mapped label string.
    """
    if label_str in Config.TARGET_LABELS:
        return label_str
    elif label_str == Config.SILENCE_LABEL:
        return label_str
    else:
        return Config.UNKNOWN_LABEL


def compute_accuracy(outputs, targets):
    """
    Computes the multiclass accuracy.

    Args:
        outputs (torch.Tensor): Raw logits of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth indices of shape (batch_size).

    Returns:
        float: The accuracy (0.0 to 1.0).
    """
    with torch.no_grad():
        # Get the index of the max log-probability
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        if total == 0:
            return 0.0
        return correct / total
