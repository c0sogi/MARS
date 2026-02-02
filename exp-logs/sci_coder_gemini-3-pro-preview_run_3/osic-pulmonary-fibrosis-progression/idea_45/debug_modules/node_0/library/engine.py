import torch
import numpy as np
from library.utils import AverageMeter, compute_metric


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, device):
    """
    Performs one epoch of training.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The BPCDS-Net model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        loss_fn (nn.Module): Loss function (LaplaceNLLLoss).
        device (torch.device): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (images, clinical_features, targets) in enumerate(loader):
        # Move data to device
        images = images.to(device)
        clinical_features = clinical_features.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model returns mu (scaled FVC) and sigma (scaled confidence)
        mu, sigma = model(images, clinical_features)

        # Calculate loss
        # Note: targets are also Z-scored in the dataset
        loss = loss_fn(mu, sigma, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def evaluate(model, loader, loss_fn, device, target_stats):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The BPCDS-Net model.
        loader (DataLoader): Validation data loader.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        target_stats (dict): Dictionary containing 'mean' and 'std' of the target variable
                             for inverse transformation.

    Returns:
        tuple: (average_loss, competition_metric_score)
    """
    model.eval()
    loss_meter = AverageMeter()

    # Store predictions and true values in original scale (ml)
    true_fvc_list = []
    pred_fvc_list = []
    pred_sigma_list = []

    # Extract stats for inverse scaling
    # FVC_ml = FVC_scaled * std + mean
    # Sigma_ml = Sigma_scaled * std
    global_mean = target_stats["mean"]
    global_std = target_stats["std"]

    with torch.no_grad():
        for images, clinical_features, targets in loader:
            images = images.to(device)
            clinical_features = clinical_features.to(device)
            targets = targets.to(device)

            # Forward pass
            mu, sigma = model(images, clinical_features)

            # Calculate validation loss on scaled data
            loss = loss_fn(mu, sigma, targets)
            loss_meter.update(loss.item(), images.size(0))

            # Inverse transform to original scale (ml) for metric calculation
            # Convert tensors to numpy
            mu_np = mu.cpu().numpy()
            sigma_np = sigma.cpu().numpy()
            target_np = targets.cpu().numpy()

            # Apply inverse transform
            # mu and target are Z-scored
            pred_fvc_ml = mu_np * global_std + global_mean
            true_fvc_ml = target_np * global_std + global_mean

            # sigma is scaled by std (since it represents uncertainty of the variable)
            pred_sigma_ml = sigma_np * global_std

            # Collect results
            true_fvc_list.extend(true_fvc_ml)
            pred_fvc_list.extend(pred_fvc_ml)
            pred_sigma_list.extend(pred_sigma_ml)

    # Compute competition metric
    # Note: compute_metric handles the clipping of sigma (max(sigma, 70)) internally
    metric_score = compute_metric(true_fvc_list, pred_fvc_list, pred_sigma_list)

    return loss_meter.avg, metric_score
