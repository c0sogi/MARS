import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("engine")


def smooth_binary_labels(targets, smoothing=0.05):
    """
    Applies label smoothing to binary targets.
    y_ls = y * (1 - alpha) + 0.5 * alpha

    Args:
        targets (torch.Tensor): Binary targets (0 or 1).
        smoothing (float): Smoothing factor alpha.

    Returns:
        torch.Tensor: Smoothed targets.
    """
    with torch.no_grad():
        targets = targets * (1.0 - smoothing) + 0.5 * smoothing
    return targets


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch using the SAM optimizer.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training dataloader.
        optimizer (SAM): The SAM optimizer instance.
        device (str): Device to train on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, angles, labels) in enumerate(dataloader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).view(-1, 1)

        # Apply Label Smoothing
        targets = smooth_binary_labels(labels, smoothing=Config.LABEL_SMOOTHING)

        # --- SAM Optimization Step ---

        # 1. First Forward Pass (Current Weights)
        # We need to compute gradients at the current point w to determine the perturbation
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # 2. First Backward Pass
        loss.backward()

        # 3. Define Closure for Second Pass (Perturbed Weights)
        # SAM will call this after applying the perturbation (ascent)
        def closure():
            # Note: SAM.first_step(zero_grad=True) clears gradients before calling this
            outputs_adv = model(images, angles)
            loss_adv = criterion(outputs_adv, targets)
            loss_adv.backward()
            return loss_adv

        # 4. SAM Step (Ascent -> Closure -> Descent)
        # This executes the perturbation, the closure (forward/backward at w+e), and the weight update
        optimizer.step(closure)

        # Zero gradients for the next batch
        optimizer.zero_grad()

        # Accumulate metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_tta(model, dataloader, device):
    """
    Validates the model using Klein Four-Group Test-Time Augmentation (TTA).
    Groups: Original, Horizontal Flip, Vertical Flip, Rotate 180.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation dataloader.
        device (str): Device to evaluate on.

    Returns:
        float: Average Log Loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Epsilon for numerical stability in Log Loss
    epsilon = 1e-7

    with torch.no_grad():
        for images, angles, labels in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).view(-1, 1)

            # TTA Generation
            # 1. Original
            img_orig = images
            # 2. Horizontal Flip (dim 3 is width)
            img_hflip = torch.flip(images, dims=[3])
            # 3. Vertical Flip (dim 2 is height)
            img_vflip = torch.flip(images, dims=[2])
            # 4. Rotate 180 (Vertical + Horizontal Flip)
            img_rot180 = torch.flip(images, dims=[2, 3])

            # Forward Passes
            logits_orig = model(img_orig, angles)
            logits_hflip = model(img_hflip, angles)
            logits_vflip = model(img_vflip, angles)
            logits_rot180 = model(img_rot180, angles)

            # Convert to Probabilities
            probs_orig = torch.sigmoid(logits_orig)
            probs_hflip = torch.sigmoid(logits_hflip)
            probs_vflip = torch.sigmoid(logits_vflip)
            probs_rot180 = torch.sigmoid(logits_rot180)

            # Average Probabilities (Invariant to cardinal symmetries)
            avg_probs = (probs_orig + probs_hflip + probs_vflip + probs_rot180) / 4.0

            # Clamp probabilities to avoid log(0)
            avg_probs_clamped = torch.clamp(avg_probs, epsilon, 1.0 - epsilon)

            # Calculate Log Loss manually
            # Loss = - [y * log(p) + (1-y) * log(1-p)]
            batch_loss = -(
                labels * torch.log(avg_probs_clamped)
                + (1 - labels) * torch.log(1 - avg_probs_clamped)
            )

            # Average over batch
            batch_loss_mean = batch_loss.mean()

            batch_size = images.size(0)
            running_loss += batch_loss_mean.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    logger.info(f"Validation Log Loss (TTA): {epoch_loss:.16f}")
    return epoch_loss


def predict_tta(model, dataloader, device):
    """
    Generates predictions for the test set using Klein Four-Group TTA.

    Args:
        model (nn.Module): The model to use for prediction.
        dataloader (DataLoader): Test dataloader.
        device (str): Device to predict on.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle both labeled (img, ang, lbl) and unlabeled (img, ang) loaders
            if len(batch) == 3:
                images, angles, _ = batch
            else:
                images, angles = batch

            images = images.to(device)
            angles = angles.to(device)

            # TTA Generation
            img_orig = images
            img_hflip = torch.flip(images, dims=[3])
            img_vflip = torch.flip(images, dims=[2])
            img_rot180 = torch.flip(images, dims=[2, 3])

            # Forward Passes
            logits_orig = model(img_orig, angles)
            logits_hflip = model(img_hflip, angles)
            logits_vflip = model(img_vflip, angles)
            logits_rot180 = model(img_rot180, angles)

            # Probabilities
            probs_orig = torch.sigmoid(logits_orig)
            probs_hflip = torch.sigmoid(logits_hflip)
            probs_vflip = torch.sigmoid(logits_vflip)
            probs_rot180 = torch.sigmoid(logits_rot180)

            # Average
            avg_probs = (probs_orig + probs_hflip + probs_vflip + probs_rot180) / 4.0

            all_probs.append(avg_probs.cpu().numpy())

    return np.concatenate(all_probs).flatten()


def update_swa_bn(swa_model, dataloader, device):
    """
    Updates BatchNorm statistics for the SWA model.
    Custom implementation required because the model takes two inputs (image, angle),
    and standard torch.optim.swa_utils.update_bn only supports single-input models.

    Args:
        swa_model (nn.Module): The averaged SWA model.
        dataloader (DataLoader): Dataloader to compute statistics on.
        device (str): Device to run on.
    """
    logger.info("Updating SWA BatchNorm statistics...")

    # Reset BN statistics
    for module in swa_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            # Set momentum to None to calculate simple cumulative average
            module.momentum = None

    swa_model.train()

    with torch.no_grad():
        for data in dataloader:
            # Handle both labeled (img, ang, lbl) and unlabeled (img, ang) loaders
            if len(data) == 3:
                images, angles, _ = data
            else:
                images, angles = data

            images = images.to(device)
            angles = angles.to(device)

            # Forward pass updates the running stats
            swa_model(images, angles)

    swa_model.eval()
    logger.info("SWA BatchNorm update complete.")


def save_submission(predictions, test_ids, output_path):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        predictions (np.ndarray): Predicted probabilities.
        test_ids (np.ndarray): Corresponding image IDs.
        output_path (str): Path to save the CSV.
    """
    df = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False, float_format="%.6f")
    logger.info(f"Submission saved to {output_path}")
