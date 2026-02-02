import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import AverageMeter, save_checkpoint


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies input-level mixup to the batch.
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
    Computes the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer:
    """
    Manages the training, validation, and SWA lifecycle.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        swa_start_epoch,
        logger,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.swa_start_epoch = swa_start_epoch
        self.logger = logger

        # Initialize SWA components
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

        self.best_auc = 0.0

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter("Loss")

        for i, (images, targets, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Apply Mixup
            images, targets_a, targets_b, lam = mixup_data(
                images, targets, Config.MIXUP_ALPHA, self.device
            )

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, model_to_eval):
        model_to_eval.eval()
        losses = AverageMeter("Val Loss")

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, targets, _ in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = model_to_eval(images)
                loss = self.criterion(outputs, targets)

                losses.update(loss.item(), images.size(0))

                # Apply Sigmoid for probabilities
                probs = torch.sigmoid(outputs)

                all_targets.append(targets.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        if len(all_targets) == 0:
            return 0.0, 0.0

        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        # Compute AUC (Macro Average)
        try:
            auc = roc_auc_score(all_targets, all_preds, average="macro")
        except ValueError:
            # Handle edge cases where a class might be missing in validation batch
            auc = 0.0

        return losses.avg, auc

    def fit(self, num_epochs, checkpoint_dir, checkpoint_prefix="model", patience=10):
        """
        Main training loop with SWA and Early Stopping logic.
        """
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)

            # Determine Phase
            is_swa_phase = epoch >= self.swa_start_epoch

            if is_swa_phase:
                # SWA Update
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                lr = self.swa_scheduler.get_last_lr()[0]
                phase_str = "SWA"
            else:
                # Standard Scheduler Step
                if self.scheduler:
                    self.scheduler.step()
                    lr = self.scheduler.get_last_lr()[0]
                else:
                    lr = self.optimizer.param_groups[0]["lr"]
                phase_str = "Standard"

            # Validate Base Model
            val_loss, val_auc = self.validate(self.model)

            self.logger.info(
                f"Epoch {epoch}/{num_epochs} [{phase_str}] - LR: {lr:.6f} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.10f}"
            )

            # Save Last Model
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "auc": val_auc,
                },
                is_best=False,
                checkpoint_dir=checkpoint_dir,
                filename=f"{checkpoint_prefix}_last.pth",
            )

            # Track Best Base Model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "auc": val_auc,
                    },
                    is_best=True,
                    checkpoint_dir=checkpoint_dir,
                    filename=f"{checkpoint_prefix}_base_best.pth",
                )
                self.logger.info(f"  * New Best Base Model AUC: {self.best_auc:.10f}")
            else:
                patience_counter += 1

            # Early Stopping (Disabled during SWA)
            if not is_swa_phase and patience_counter >= patience:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch} (Pre-SWA)."
                )
                break

        # End of Training: Finalize SWA
        self.logger.info("Training finished. Finalizing SWA Model...")

        # Update BN statistics for SWA model
        self.logger.info("Updating SWA Batch Norm statistics...")
        update_bn(self.train_loader, self.swa_model, device=self.device)

        # Validate SWA Model
        swa_val_loss, swa_val_auc = self.validate(self.swa_model)
        self.logger.info(
            f"SWA Model Final Result - Val Loss: {swa_val_loss:.6f} - Val AUC: {swa_val_auc:.10f}"
        )

        # Save SWA Model
        save_checkpoint(
            {
                "epoch": num_epochs,
                "state_dict": self.swa_model.state_dict(),
                "auc": swa_val_auc,
            },
            is_best=False,
            checkpoint_dir=checkpoint_dir,
            filename=f"{checkpoint_prefix}_swa.pth",
        )

        return self.best_auc, swa_val_auc
