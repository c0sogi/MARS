import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import AveragedModel

from library.config import Config
from library.utils import AverageMeter, calculate_roc_auc, get_logger
from library.data import mixup_data

# Initialize logger
logger = get_logger(name="Engine")


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA) for the model.
    """

    def __init__(self, model, device):
        self.device = device
        self.swa_model = AveragedModel(model).to(device)
        self.start_epoch = Config.SWA_START_EPOCH
        self.enabled = False

    def update(self, model, epoch):
        """
        Updates the SWA model parameters if the current epoch is past the start epoch.
        """
        if epoch >= self.start_epoch:
            self.enabled = True
            self.swa_model.update_parameters(model)

    def update_bn(self, loader):
        """
        Updates Batch Normalization statistics for the SWA model.
        Custom implementation to handle dictionary-yielding dataloaders.
        """
        if not self.enabled:
            return

        logger.info("Updating SWA Batch Normalization statistics...")
        self.swa_model.train()

        with torch.no_grad():
            for i, batch in enumerate(loader):
                # Extract image tensor from the batch dictionary
                if isinstance(batch, dict):
                    input_var = batch["image"].to(self.device)
                else:
                    # Fallback if loader changes
                    input_var = batch[0].to(self.device)

                # Forward pass to update BN stats
                self.swa_model(input_var)

    def get_model(self):
        return self.swa_model


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup and Multi-Task Loss.
    """
    model.train()

    losses = AverageMeter()
    class_losses = AverageMeter()
    qual_losses = AverageMeter()

    # Loss functions
    # Note: reduction='mean' is standard, but with mixup we calculate per-sample weighted sum then mean
    bce_crit = nn.BCEWithLogitsLoss()
    mse_crit = nn.MSELoss()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        qualities = batch["quality"].to(device)

        # Apply Mixup
        mixed_images, y_class_a, y_class_b, y_qual_a, y_qual_b, lam = mixup_data(
            images, labels, qualities, alpha=Config.MIXUP_ALPHA, device=device
        )

        # Forward pass
        outputs = model(mixed_images)
        pred_class = outputs["class"].view(-1)
        pred_qual = outputs["quality"].view(-1)

        # Calculate Mixup Loss for Classification
        loss_class = lam * bce_crit(pred_class, y_class_a) + (1 - lam) * bce_crit(
            pred_class, y_class_b
        )

        # Calculate Mixup Loss for Quality Regression
        loss_qual = lam * mse_crit(pred_qual, y_qual_a) + (1 - lam) * mse_crit(
            pred_qual, y_qual_b
        )

        # Total Weighted Loss
        loss = loss_class + Config.AUX_WEIGHT * loss_qual

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        class_losses.update(loss_class.item(), images.size(0))
        qual_losses.update(loss_qual.item(), images.size(0))

    return {
        "loss": losses.avg,
        "class_loss": class_losses.avg,
        "qual_loss": qual_losses.avg,
    }


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter()
    class_losses = AverageMeter()
    qual_losses = AverageMeter()

    # Containers for AUC calculation
    all_preds = []
    all_targets = []

    bce_crit = nn.BCEWithLogitsLoss()
    mse_crit = nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            qualities = batch["quality"].to(device)

            outputs = model(images)
            pred_class_logits = outputs["class"].view(-1)
            pred_qual = outputs["quality"].view(-1)

            # Calculate Losses (No Mixup)
            loss_class = bce_crit(pred_class_logits, labels)
            loss_qual = mse_crit(pred_qual, qualities)
            loss = loss_class + Config.AUX_WEIGHT * loss_qual

            # Update metrics
            losses.update(loss.item(), images.size(0))
            class_losses.update(loss_class.item(), images.size(0))
            qual_losses.update(loss_qual.item(), images.size(0))

            # Store predictions for AUC
            probs = torch.sigmoid(pred_class_logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Calculate AUC
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    auc_score = calculate_roc_auc(all_targets, all_preds)

    return {
        "loss": losses.avg,
        "class_loss": class_losses.avg,
        "qual_loss": qual_losses.avg,
        "auc": auc_score,
    }


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=Config.EPOCHS,
    patience=8,
):
    """
    Main training loop with SWA and Early Stopping.
    """
    best_auc = 0.0
    best_model_state = None
    early_stop_counter = 0

    # Initialize SWA
    swa_handler = SWAHandler(model, device)

    logger.info(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Update SWA
        swa_handler.update(model, epoch)

        # Logging
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch}/{epochs} [{elapsed:.1f}s] | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val AUC: {val_metrics['auc']:.8f} | "
            f"Val Qual MSE: {val_metrics['qual_loss']:.6f}"
        )

        # Checkpointing (Best Model)
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0

            # Save best model
            save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(best_model_state, save_path)
        else:
            early_stop_counter += 1

        # Early Stopping
        # We disable early stopping if we are close to or inside the SWA phase
        # to ensure we collect enough snapshots.
        if early_stop_counter >= patience:
            if epoch < Config.SWA_START_EPOCH - 2:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
            else:
                logger.info(
                    f"Early stopping criteria met, but continuing for SWA collection."
                )

    # Finalize Training
    logger.info("Training complete.")

    # Reload best model for final non-SWA state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Finalize SWA if active
    if swa_handler.enabled:
        logger.info("Finalizing SWA model...")
        # Update BN statistics using training data
        swa_handler.update_bn(train_loader)

        # Save SWA model
        swa_save_path = os.path.join(Config.CHECKPOINT_DIR, "swa_model.pth")
        torch.save(swa_handler.get_model().state_dict(), swa_save_path)
        logger.info(f"SWA model saved to {swa_save_path}")

    return best_auc
