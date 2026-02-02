import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import AveragedModel
from library.utils import calculate_log_loss


def apply_label_smoothing(targets, smoothing=0.0):
    """
    Applies label smoothing to binary targets.
    New target = target * (1 - smoothing) + 0.5 * smoothing
    """
    if smoothing == 0.0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def train_one_epoch(model, loader, optimizer, device, epoch, label_smoothing=0.0):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        device: Torch device.
        epoch: Current epoch number (for logging, optional).
        label_smoothing: Float, smoothing factor for binary labels.

    Returns:
        avg_loss: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    # BCEWithLogitsLoss combines Sigmoid and BCE.
    # It does not support label_smoothing arg in older PyTorch versions,
    # so we smooth targets manually.
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, angles, targets, _) in enumerate(loader):
        images = images.to(device)
        angles = angles.to(device)
        # Ensure targets are float and shape (B, 1)
        targets = targets.to(device).float().view(-1, 1)

        # Apply label smoothing
        smoothed_targets = apply_label_smoothing(targets, label_smoothing)

        optimizer.zero_grad()

        # Forward pass: model expects (x, inc_angle)
        logits = model(images, angles)
        loss = criterion(logits, smoothed_targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    avg_loss = running_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss


def validate_tta(model, loader, device):
    """
    Validates the model using Klein Four-Group TTA (Original, HFlip, VFlip, Rot180).

    Args:
        model: The neural network.
        loader: DataLoader for validation data.
        device: Torch device.

    Returns:
        loss: The log loss calculated on the validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, targets, _ in loader:
            images = images.to(device)
            angles = angles.to(device)

            # TTA 1: Original
            logits1 = model(images, angles)
            probs1 = torch.sigmoid(logits1)

            # TTA 2: Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            logits2 = model(images_h, angles)
            probs2 = torch.sigmoid(logits2)

            # TTA 3: Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            logits3 = model(images_v, angles)
            probs3 = torch.sigmoid(logits3)

            # TTA 4: Rotate 180 (H + V)
            images_hv = torch.flip(images, [2, 3])
            logits4 = model(images_hv, angles)
            probs4 = torch.sigmoid(logits4)

            # Average probabilities
            avg_probs = (probs1 + probs2 + probs3 + probs4) / 4.0

            all_preds.append(avg_probs.cpu().numpy())
            all_targets.append(targets.numpy())

    if len(all_preds) == 0:
        return 0.0

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    loss = calculate_log_loss(all_targets, all_preds)
    return loss


def run_swa_phase(
    model, loader, optimizer, device, swa_epochs, swa_lr, label_smoothing=0.0
):
    """
    Runs the SWA (Stochastic Weight Averaging) phase.

    Args:
        model: The base model (already trained).
        loader: Training dataloader.
        optimizer: Optimizer.
        device: Device.
        swa_epochs: Number of epochs to run SWA.
        swa_lr: Constant learning rate for SWA.
        label_smoothing: Label smoothing factor.

    Returns:
        swa_model: The averaged model with updated Batch Norm statistics.
    """
    print(f"Starting SWA Phase for {swa_epochs} epochs with LR={swa_lr}...")

    # Initialize Averaged Model
    swa_model = AveragedModel(model)

    # Set constant learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = swa_lr

    for epoch in range(swa_epochs):
        # Train one epoch on base model
        loss = train_one_epoch(model, loader, optimizer, device, epoch, label_smoothing)

        # Update SWA parameters
        swa_model.update_parameters(model)

        print(f"SWA Epoch {epoch+1}/{swa_epochs} - Train Loss: {loss:.6f}")

    # Update Batch Normalization statistics
    # Since our model takes multiple inputs, standard torch.optim.swa_utils.update_bn won't work easily.
    # We implement the update logic manually.
    print("Updating SWA Batch Normalization statistics...")

    swa_model.train()

    # Reset BN stats and set momentum to None for cumulative moving average (exact stats)
    for module in swa_model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            module.momentum = None
            module.num_batches_tracked = torch.tensor(
                0, dtype=torch.long, device=device
            )

    with torch.no_grad():
        for images, angles, _, _ in loader:
            images = images.to(device)
            angles = angles.to(device)
            # Forward pass updates running stats
            _ = swa_model(images, angles)

    return swa_model


def predict_ensemble(models, loader, device):
    """
    Generates predictions using an ensemble of models with TTA.

    Args:
        models: List of models (can be regular or SWA models).
        loader: Test dataloader.
        device: Device.

    Returns:
        all_ids: List of image IDs.
        ensemble_probs: Numpy array of predicted probabilities.
    """
    # Set all models to eval
    for m in models:
        m.eval()
        m.to(device)

    all_ids = []
    ensemble_probs = []

    with torch.no_grad():
        for batch_data in loader:
            # Test loader yields (images, angles, ids)
            images, angles, ids = batch_data
            images = images.to(device)
            angles = angles.to(device)

            all_ids.extend(ids)

            # Prepare TTA inputs
            images_h = torch.flip(images, [3])
            images_v = torch.flip(images, [2])
            images_hv = torch.flip(images, [2, 3])

            batch_preds = []

            for model in models:
                # 1. Original
                p1 = torch.sigmoid(model(images, angles))
                # 2. HFlip
                p2 = torch.sigmoid(model(images_h, angles))
                # 3. VFlip
                p3 = torch.sigmoid(model(images_v, angles))
                # 4. Rot180
                p4 = torch.sigmoid(model(images_hv, angles))

                # Average for this model
                avg_p = (p1 + p2 + p3 + p4) / 4.0
                batch_preds.append(avg_p.cpu().numpy())

            # Average across models for this batch
            # batch_preds shape: (num_models, batch_size, 1)
            batch_preds = np.array(batch_preds)
            mean_preds = np.mean(batch_preds, axis=0)  # (batch_size, 1)
            ensemble_probs.append(mean_preds)

    if len(ensemble_probs) > 0:
        ensemble_probs = np.concatenate(ensemble_probs).flatten()
    else:
        ensemble_probs = np.array([])

    return all_ids, ensemble_probs
