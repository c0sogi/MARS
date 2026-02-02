import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import update_bn as torch_update_bn
from library.utils import AverageMeter, compute_roc_auc
from library.config import Config


def train_one_epoch(
    model, loader, optimizer, device, epoch, mixup_active=False, alpha=0.2
):
    """
    Trains the model for one epoch, optionally using Manifold Mixup.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        device (str): Device to run training on.
        epoch (int): Current epoch number.
        mixup_active (bool): Whether to apply Manifold Mixup.
        alpha (float): Beta distribution parameter for mixup.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if mixup_active:
            # Forward pass with Manifold Mixup
            # Model returns: logits, target_a, target_b, lam
            logits, target_a, target_b, lam = model(
                images, target=targets, mixup=True, alpha=alpha
            )

            # Mixup Loss
            loss = lam * criterion(logits, target_a) + (1 - lam) * criterion(
                logits, target_b
            )
        else:
            # Standard Forward pass
            logits = model(images, target=None, mixup=False)
            loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, macro_roc_auc_score)
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            # Standard forward pass (no mixup during validation)
            logits = model(images, target=None, mixup=False)
            loss = criterion(logits, targets)

            # Apply sigmoid for metric calculation
            probs = torch.sigmoid(logits)

            losses.update(loss.item(), images.size(0))

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    score = compute_roc_auc(all_targets, all_preds)

    return losses.avg, score


def generate_predictions(model, loader, device, use_tta=True):
    """
    Generates predictions for a dataset, optionally using Test-Time Augmentation (TTA).

    TTA Strategy: Average of (Original Image, Horizontally Flipped Image).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for inference (Test set).
        device (str): Device to run inference on.
        use_tta (bool): Whether to apply horizontal flip TTA.

    Returns:
        dict: Mapping from rec_id (int) to predicted probabilities (np.array).
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for images, _, rec_ids in loader:
            images = images.to(device)

            # 1. Forward pass on original images
            logits_orig = model(images, target=None, mixup=False)
            probs_orig = torch.sigmoid(logits_orig)

            if use_tta:
                # 2. Forward pass on horizontally flipped images
                # Dim 3 is width (N, C, H, W)
                images_flipped = torch.flip(images, dims=[3])
                logits_flip = model(images_flipped, target=None, mixup=False)
                probs_flip = torch.sigmoid(logits_flip)

                # Average probabilities
                probs_final = (probs_orig + probs_flip) / 2.0
            else:
                probs_final = probs_orig

            # Store results
            probs_np = probs_final.cpu().numpy()
            rec_ids_np = rec_ids.numpy()

            for i in range(len(rec_ids_np)):
                results[int(rec_ids_np[i])] = probs_np[i]

    return results


def update_bn(loader, model, device):
    """
    Updates BatchNorm statistics for the SWA model.
    Wrapper around torch.optim.swa_utils.update_bn to handle custom logic if needed.

    Args:
        loader (DataLoader): Training loader to compute statistics.
        model (nn.Module): The SWA model.
        device (str): Device.
    """
    model.train()
    # We use the standard torch utility, ensuring data is on the correct device
    # The utility expects the loader to yield samples, or (sample, label).
    # Our loader yields (images, targets, rec_ids).
    # update_bn handles (x, y) but might trip on (x, y, z).
    # We create a simple generator wrapper to yield only images.

    def _loader_wrapper(dataloader):
        for data in dataloader:
            # data is [images, targets, rec_ids]
            yield data[0].to(device)

    # Reset BN running stats
    # (Optional but recommended for SWA to clear old stats)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            module.momentum = None  # Use simple average

    torch_update_bn(_loader_wrapper(loader), model, device=device)
