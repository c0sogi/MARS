import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Also configures cuDNN for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
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


def save_checkpoint(state, checkpoint_dir, filename):
    """
    Saves the model checkpoint (state dictionary) to the specified directory.

    Args:
        state (dict): The model state dictionary and other metadata.
        checkpoint_dir (str): The directory to save the checkpoint in.
        filename (str): The name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Performs Mixup augmentation on the input batch.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup interpolation coefficient parameter.
        device (str): Device to perform computations on.

    Returns:
        mixed_x (torch.Tensor): Mixed input images.
        y_a (torch.Tensor): Targets for the first image set.
        y_b (torch.Tensor): Targets for the second image set.
        lam (float): The mixing coefficient lambda.
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
    Computes the Mixup loss.

    Args:
        criterion (callable): The loss function.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Targets for the first image set.
        y_b (torch.Tensor): Targets for the second image set.
        lam (float): The mixing coefficient lambda.

    Returns:
        loss (torch.Tensor): The weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
