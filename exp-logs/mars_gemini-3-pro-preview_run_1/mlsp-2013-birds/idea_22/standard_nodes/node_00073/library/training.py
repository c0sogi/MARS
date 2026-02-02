import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, update_bn

from library import config, utils, data


def train_one_epoch(loader, model, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        # Apply Mixup
        mixed_images, targets_a, targets_b, lam = data.mixup_data(
            images, targets, alpha=config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        outputs = model(mixed_images)

        # Mixup Loss
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
            outputs, targets_b
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and macro ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Collect for metrics
            probs = torch.sigmoid(outputs)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)

    # Calculate ROC AUC
    # Handle edge cases where a class might not be present in the validation set
    # Cite debug_lesson_5: Safeguard Global Metrics Against Degenerate Data Subsets
    aucs = []
    for i in range(all_targets.shape[1]):
        # Only calculate AUC if both classes (0 and 1) are present
        if len(np.unique(all_targets[:, i])) > 1:
            score = roc_auc_score(all_targets[:, i], all_probs[:, i])
            aucs.append(score)

    # If no classes are valid (e.g. extremely small debug set), return 0.5 (random guess)
    val_auc = np.mean(aucs) if aucs else 0.5

    return epoch_loss, val_auc


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        checkpoint_dir,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.criterion = nn.BCEWithLogitsLoss()

        # SWA Components
        self.swa_model = AveragedModel(model)
        self.swa_start = config.SWA_START_EPOCH

        self.best_auc = 0.0
        self.logger = utils.get_logger("Trainer")

    def fit(self, epochs):
        self.logger.info(
            f"Starting training for {epochs} epochs on device {self.device}"
        )

        for epoch in range(epochs):
            # --- SWA Logic: Switch LR if entering SWA phase ---
            if epoch == self.swa_start:
                self.logger.info(
                    f"Epoch {epoch}: Starting SWA Phase. Switching LR to {config.SWA_LR}"
                )
                # Update optimizer LR for all param groups
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = config.SWA_LR

            # --- Training Step ---
            train_loss = train_one_epoch(
                self.train_loader,
                self.model,
                self.optimizer,
                self.criterion,
                self.device,
                epoch,
            )

            # --- SWA Update ---
            if epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)

            # --- Validation Step ---
            # Note: We validate the current model, not the SWA model during training
            val_loss, val_auc = validate(
                self.val_loader, self.model, self.criterion, self.device
            )

            # --- Scheduler Step (Only before SWA) ---
            if epoch < self.swa_start and self.scheduler is not None:
                self.scheduler.step()

            # --- Logging ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"LR: {current_lr:.6f} - "
                f"Train Loss: {train_loss:.10f} - "
                f"Val Loss: {val_loss:.10f} - "
                f"Val AUC: {val_auc:.10f}"
            )

            # --- Checkpointing ---
            # Save Last
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "val_auc": val_auc,
                },
                is_best=False,
                checkpoint_dir=self.checkpoint_dir,
                filename="model_last.pth",
            )

            # Save Best (Only relevant for non-SWA phase or tracking standard model performance)
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                utils.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "val_auc": val_auc,
                    },
                    is_best=True,
                    checkpoint_dir=self.checkpoint_dir,
                    filename="model_best.pth",
                )
                self.logger.info(f"New best model found! AUC: {val_auc:.10f}")

        # --- End of Training: Finalize SWA ---
        self.logger.info("Training complete. Finalizing SWA model...")

        # Update BN statistics for the SWA model
        # We need to perform a pass over the train loader to calculate mean/var for BN layers
        # The SWA model is currently on CPU or device depending on AveragedModel defaults, usually same as model
        # Ensure it is on the correct device
        # AveragedModel usually keeps weights on the device of the model passed in init.

        # Note: update_bn expects the model to be on the device where data is.
        # We need to ensure the swa_model is on self.device
        # AveragedModel doesn't have a simple .to() method that moves internal buffers easily in older pytorch versions,
        # but usually it works if the base model was on device.

        # We use the train_loader but without augmentation/mixup ideally, or just standard train loader.
        # Standard practice is to use the train loader.

        update_bn(self.train_loader, self.swa_model, device=self.device)

        # Save SWA Model
        # We save the underlying module state dict
        utils.save_checkpoint(
            {
                "epoch": epochs,
                "state_dict": self.swa_model.module.state_dict(),
                "val_auc": 0.0,  # SWA AUC not computed yet
            },
            is_best=False,
            checkpoint_dir=self.checkpoint_dir,
            filename="model_swa.pth",
        )
        self.logger.info(
            f"SWA model saved to {os.path.join(self.checkpoint_dir, 'model_swa.pth')}"
        )
