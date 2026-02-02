import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, update_bn
from library.configuration import Config
from library.utilities import AverageMeter


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    """
    Trains the model for one epoch using Geometric Consistency Regularization.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training dataloader returning ((img1, img2), angle, label).
        optimizer (Optimizer): The optimizer.
        loss_fn (nn.Module): The ConsistencyLoss function.
        device (str): Device to run on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, angles, labels) in enumerate(loader):
        # Unpack dual views
        img1, img2 = images

        # Move to device
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward passes
        logits1 = model(img1, angles)
        logits2 = model(img2, angles)

        # Compute Consistency Loss
        loss = loss_fn(logits1, logits2, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), img1.size(0))

    print(f"Epoch {epoch} Train Loss: {losses.avg}")
    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Log Loss.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation dataloader returning (img, angle, label).
        device (str): Device to run on.

    Returns:
        float: Average Log Loss (BCE).
    """
    model.eval()
    losses = AverageMeter()

    # Standard Log Loss for validation metric
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            # Forward pass
            logits = model(images, angles)

            # Compute Loss
            loss = criterion(logits, labels)

            # Update metrics
            losses.update(loss.item(), images.size(0))

    print(f"Validation Log Loss: {losses.avg}")
    return losses.avg


def swa_step(swa_model, model):
    """
    Updates the SWA model parameters.

    Args:
        swa_model (AveragedModel): The SWA model wrapper.
        model (nn.Module): The current training model.
    """
    swa_model.update_parameters(model)


def update_swa_batch_norm(swa_model, loader, device):
    """
    Updates Batch Normalization statistics for the SWA model.

    Args:
        swa_model (AveragedModel): The SWA model.
        loader (DataLoader): Dataloader to compute stats on.
        device (str): Device.
    """
    # Create a wrapper to feed (img, angle) to the model during update_bn
    # update_bn expects a loader that yields input data.
    # Since our model takes two inputs (x, inc_angle), we need a custom forward
    # or a custom loader adapter. However, torch.optim.swa_utils.update_bn
    # simply calls model(input) for items in loader.
    # We need to ensure the loader yields the correct input structure or handle it.

    # Standard update_bn assumes loader yields x.
    # Our loader yields (x, angle, y).
    # We can write a custom loop or a generator.

    swa_model.train()

    # We manually implement the BN update logic to handle the dual input signature
    # based on the logic within torch.optim.swa_utils.update_bn

    momenta = {}
    for module in swa_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    with torch.no_grad():
        for images, angles, _ in loader:
            # Handle dual-view tuple if passed from train loader,
            # though usually we use a clean loader for BN update.
            if isinstance(images, (list, tuple)):
                images = images[0]  # Take first view

            images = images.to(device)
            angles = angles.to(device)

            # Forward pass updates running stats
            swa_model(images, angles)

    for module in swa_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]
