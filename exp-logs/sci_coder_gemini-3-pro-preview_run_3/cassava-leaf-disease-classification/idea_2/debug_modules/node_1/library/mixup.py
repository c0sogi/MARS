import numpy as np
import torch


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.

    Args:
        size (torch.Size): Size of the input tensor (Batch, Channel, Height, Width).
        lam (float): Lambda value sampled from Beta distribution.

    Returns:
        tuple: (bbx1, bby1, bbx2, bby2) coordinates.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniformly sample the center of the box
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1.0):
    """
    Performs MixUp augmentation.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Input labels.
        alpha (float): Alpha parameter for Beta distribution.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Labels for the first image set.
        y_b (torch.Tensor): Labels for the second image set.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    Performs CutMix augmentation.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Input labels.
        alpha (float): Alpha parameter for Beta distribution.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Labels for the first image set.
        y_b (torch.Tensor): Labels for the second image set.
        lam (float): Adjusted mixing coefficient based on bbox area.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Clone to avoid modifying the original tensor in-place
    mixed_x = x.clone()

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    # Paste the patch from the shuffled batch
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to match the exact pixel ratio of the patch
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
