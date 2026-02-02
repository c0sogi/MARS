import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.metrics import log_loss, accuracy_score

import library.config as config
from library.utils import save_checkpoint

# =============================================================================
# LOSS FUNCTION
# =============================================================================


class BCEWithLogitsLossLabelSmoothing(nn.Module):
    """
    Binary Cross Entropy with Logits and Label Smoothing.
    Formula: y_smooth = y * (1 - epsilon) + 0.5 * epsilon
    """

    def __init__(self, epsilon=config.LABEL_SMOOTHING):
        super(BCEWithLogitsLossLabelSmoothing, self).__init__()
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # targets shape: (B, 1)
        targets_smooth = targets * (1 - self.epsilon) + 0.5 * self.epsilon
        return self.bce(logits, targets_smooth)


# =============================================================================
# CORE TRAINING FUNCTIONS
# =============================================================================


def train_one_epoch(loader, model, optimizer, criterion, device):
    """
    Performs one epoch of training using the SAM optimizer.
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)

        batch_size = images.size(0)
        total_samples += batch_size

        # Container to capture logits from the first forward pass of SAM
        logits_container = []

        # SAM Closure: Computes loss and gradients
        def closure():
            # Note: We do not zero_grad here because SAM handles it internally
            # via first_step(zero_grad=True) for the second pass.
            # The outer loop ensures the initial gradients are zeroed.

            output = model(images, angles)
            loss = criterion(output, labels)
            loss.backward()

            # Capture logits from the first pass (at current weights) for metrics
            if not logits_container:
                logits_container.append(output.detach())

            return loss

        # Ensure gradients are zero before starting the SAM step
        optimizer.zero_grad()

        # SAM Step (performs dual forward/backward passes)
        loss = optimizer.step(closure)

        # Metrics
        running_loss += loss.item() * batch_size

        # Calculate accuracy from the first forward pass
        preds = torch.sigmoid(logits_container[0]) > 0.5
        running_corrects += torch.sum(preds == (labels > 0.5)).item()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def validate_tta(loader, model, criterion, device):
    """
    Validates the model using Klein Four-Group Test-Time Augmentation (TTA).
    Augmentations: Original, H-Flip, V-Flip, Rotate180.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    # Standard BCELoss for probabilities (since we average probabilities)
    # We use the raw criterion passed in, but if it expects logits, we must be careful.
    # The criterion passed is usually BCEWithLogitsLoss.
    # To compute validation loss correctly with TTA, we average probabilities then take log loss.
    # We will use sklearn's log_loss for the final metric to be robust.

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            # 1. Original
            out1 = torch.sigmoid(model(images, angles))

            # 2. Horizontal Flip (dim 3 is width)
            out2 = torch.sigmoid(model(torch.flip(images, [3]), angles))

            # 3. Vertical Flip (dim 2 is height)
            out3 = torch.sigmoid(model(torch.flip(images, [2]), angles))

            # 4. Rotate 180 (equivalent to flip V + flip H, or rot90 k=2)
            out4 = torch.sigmoid(model(torch.rot90(images, 2, [2, 3]), angles))

            # Average Probabilities
            avg_preds = (out1 + out2 + out3 + out4) / 4.0

            # Store for metrics
            all_preds.append(avg_preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute Metrics
    # Clip predictions to avoid log(0)
    eps = 1e-15
    all_preds_clipped = np.clip(all_preds, eps, 1 - eps)

    val_loss = log_loss(all_targets, all_preds_clipped)
    val_acc = accuracy_score(all_targets, (all_preds > 0.5).astype(int))

    return val_loss, val_acc


def predict_tta(loader, model, device):
    """
    Generates predictions for the test set using TTA.
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            # TTA Forward Passes
            out1 = torch.sigmoid(model(images, angles))
            out2 = torch.sigmoid(model(torch.flip(images, [3]), angles))
            out3 = torch.sigmoid(model(torch.flip(images, [2]), angles))
            out4 = torch.sigmoid(model(torch.rot90(images, 2, [2, 3]), angles))

            avg_preds = (out1 + out2 + out3 + out4) / 4.0

            preds_list.append(avg_preds.cpu().numpy())
            ids_list.extend(ids)

    return np.concatenate(preds_list), np.array(ids_list)


def update_swa_bn_stats(loader, swa_model, device):
    """
    Wrapper to update Batch Normalization statistics for the SWA model.
    Adapts the loader to yield only inputs (images, angles) as expected by update_bn.
    """
    # update_bn expects a loader that yields input tensors.
    # Our loader yields (images, angles, labels) or (images, angles, ids).
    # We need a custom generator or wrapper.

    # We define a helper function that performs the forward pass logic required by update_bn
    # However, torch.optim.swa_utils.update_bn simply calls model(input) during the loop.
    # Since our model takes two arguments (x, angle), standard update_bn might fail if it unpacks *input.
    # Standard update_bn implementation:
    #   for input in loader:
    #       input = input.to(device)
    #       model(input)
    #
    # If loader yields a tuple, it does model(tuple) which is wrong.
    # We must implement a custom update_bn loop.

    swa_model.train()
    with torch.no_grad():
        for batch in loader:
            # Handle both train (3 items) and test (3 items) loaders
            if len(batch) == 3:
                images, angles, _ = batch
            else:
                images, angles = batch  # Should not happen with our dataset

            images = images.to(device)
            angles = angles.to(device)

            # Forward pass updates BN statistics
            swa_model(images, angles)


# =============================================================================
# ORCHESTRATOR
# =============================================================================


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience=config.EARLY_STOPPING_PATIENCE,
    use_swa=False,
    swa_start_epoch=None,
    save_dir=config.CHECKPOINT_DIR,
    fold_idx=0,
):
    """
    Orchestrates the training process, including Phase 1 (SAM) and Phase 2 (SWA).

    Args:
        use_swa (bool): If True, enables SWA transition after swa_start_epoch.
        swa_start_epoch (int): Epoch to start SWA (Phase 2).
    """
    criterion = BCEWithLogitsLossLabelSmoothing()

    # Setup SWA
    swa_model = None
    if use_swa:
        swa_model = AveragedModel(model)

    best_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training on device: {device}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- Training Step ---
        # If in SWA phase, we still train using SAM, but we update SWA model afterwards
        train_loss, train_acc = train_one_epoch(
            train_loader, model, optimizer, criterion, device
        )

        # --- SWA Update ---
        in_swa_phase = (
            use_swa and (swa_start_epoch is not None) and (epoch >= swa_start_epoch)
        )
        if in_swa_phase:
            swa_model.update_parameters(model)
            # We do NOT step the scheduler in SWA phase if we want constant LR,
            # or we step a specific SWA scheduler.
            # The idea says: "Set SWA learning rate to be equal to LR_final".
            # Assuming the scheduler has decayed to that point, we just keep it there.
        else:
            # Step scheduler in Phase 1
            if scheduler is not None:
                # ReduceLROnPlateau expects a metric. We use val_loss if available, else train_loss.
                # But we step after validation.
                pass

        # --- Validation Step ---
        val_loss = 0.0
        val_acc = 0.0
        if val_loader is not None:
            # If in SWA phase, we should validate the SWA model (after BN update),
            # but updating BN every epoch is expensive.
            # Usually we validate the base model during training and SWA model only at the end.
            # We will validate the current base model.
            val_loss, val_acc = validate_tta(val_loader, model, criterion, device)

            # Scheduler Step
            if scheduler is not None and not in_swa_phase:
                scheduler.step(val_loss)
        else:
            # If no validation (Full Train Mode), we just step scheduler on train loss if needed
            if scheduler is not None and not in_swa_phase:
                scheduler.step(train_loss)

        elapsed = time.time() - start_time

        # --- Logging ---
        log_msg = (
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f}"
        )

        if val_loader is not None:
            log_msg += f" | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"

        if in_swa_phase:
            log_msg += " [SWA Active]"

        print(log_msg + f" | Time: {elapsed:.1f}s")

        # --- Checkpointing & Early Stopping ---
        # Only applies in Phase 1 (Calibration) or if we are tracking best model
        if not in_swa_phase and val_loader is not None:
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    {"state_dict": model.state_dict(), "epoch": epoch},
                    is_best=True,
                    checkpoint_dir=save_dir,
                    filename=f"model_fold{fold_idx}.pth",
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

    # --- Finalize ---
    if use_swa and swa_model is not None:
        print("Finalizing SWA: Updating BN statistics...")
        update_swa_bn_stats(train_loader, swa_model, device)

        # Save SWA model
        save_checkpoint(
            {
                "state_dict": swa_model.module.state_dict(),
                "epoch": epochs,
            },  # Unwrap AveragedModel
            is_best=False,
            checkpoint_dir=save_dir,
            filename=f"swa_model_{fold_idx}.pth",
        )
        return swa_model.module

    return model
