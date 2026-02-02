import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library import utils
from library.model import SEResNet34


class Trainer:
    """
    Trainer class for the Iterative Attentive SWA-Distillation pipeline.
    Handles training, validation, Mixup, SWA, and checkpointing.
    """

    def __init__(self, train_loader, val_loader, device=None, logger=None):
        """
        Initialize the Trainer.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device, optional): Device to train on.
            logger (logging.Logger, optional): Logger instance.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else utils.get_device()
        self.logger = logger if logger else utils.get_logger()

        # Initialize Model
        self.model = SEResNet34(pretrained=Config.PRETRAINED)
        self.model.to(self.device)

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # SWA Configuration
        self.swa_start_epoch = Config.SWA_START_EPOCH
        self.epochs = Config.EPOCHS

        # Schedulers
        # Phase 1: Cosine Annealing until SWA start
        # We set T_max to swa_start_epoch so it reaches min LR right before SWA starts
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.swa_start_epoch, eta_min=1e-6
        )

        # Phase 2: SWA Scheduler
        # Keeps LR constant or anneals cyclically. Here we use constant SWA_LR.
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=Config.SWA_LR)

        # Metrics tracking
        self.best_auc = 0.0
        self.best_epoch = 0

        # Early Stopping
        self.patience = 10
        self.counter = 0

    def mixup_data(self, x, y, alpha=Config.MIXUP_ALPHA):
        """
        Applies Mixup augmentation to the batch.
        Returns mixed inputs, pairs of targets, and lambda.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        """
        Calculates loss for mixed inputs.
        """
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = utils.AverageMeter()
        start_time = time.time()

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup
            images, labels_a, labels_b, lam = self.mixup_data(images, labels)

            # Forward pass
            outputs = self.model(images)
            loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        elapsed = time.time() - start_time

        # Scheduler Step Logic
        if epoch >= self.swa_start_epoch:
            # SWA Phase: Update averaged model and use SWA scheduler
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
            lr = self.swa_scheduler.get_last_lr()[0]
            mode = "SWA"
        else:
            # Normal Phase: Use Cosine scheduler
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            mode = "Normal"

        self.logger.info(
            f"Epoch {epoch} [{mode}] | Time: {elapsed:.2f}s | Loss: {losses.avg:.6f} | LR: {lr:.8f}"
        )
        return losses.avg

    def validate(self, model_to_validate):
        """
        Runs validation on the provided model.
        """
        model_to_validate.eval()
        losses = utils.AverageMeter()

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model_to_validate(images)
                loss = self.criterion(outputs, labels)

                losses.update(loss.item(), images.size(0))

                # Apply sigmoid for AUC calculation
                probs = torch.sigmoid(outputs)

                all_targets.append(labels.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        # Concatenate results
        if len(all_targets) > 0:
            all_targets = np.concatenate(all_targets)
            all_preds = np.concatenate(all_preds)
            auc = utils.calculate_auc(all_targets, all_preds)
        else:
            auc = 0.0

        return losses.avg, auc

    def run(self, save_name="model"):
        """
        Executes the full training pipeline.

        Args:
            save_name (str): Prefix for saved checkpoints (e.g., 'teacher_0').

        Returns:
            nn.Module: The best model (or SWA model) after training.
            float: The best AUC score.
        """
        self.logger.info(
            f"Starting training for {self.epochs} epochs on {self.device}..."
        )
        self.logger.info(f"SWA will start at epoch {self.swa_start_epoch}.")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)

            # Validate the base model to track progress
            val_loss, val_auc = self.validate(self.model)

            self.logger.info(
                f"Epoch {epoch} Validation | Loss: {val_loss:.15f} | AUC: {val_auc:.15f}"
            )

            # Save Best Base Model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_epoch = epoch
                self.counter = 0  # Reset early stopping counter

                base_save_path = os.path.join(
                    Config.WORKING_DIR, f"{save_name}_base_best.pth"
                )
                utils.save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "best_auc": self.best_auc,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    is_best=False,
                    filepath=base_save_path,
                )
            else:
                # Early Stopping Logic (Only active before SWA)
                if epoch < self.swa_start_epoch:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                        break

        # Finalize SWA if applicable
        if self.epochs >= self.swa_start_epoch:
            self.logger.info("Finalizing SWA model (updating BatchNorm statistics)...")
            # update_bn expects a loader and the model.
            # It updates the running_mean/_var buffers in the model's BN layers.
            update_bn(self.train_loader, self.swa_model, device=self.device)

            # Validate SWA model
            swa_val_loss, swa_val_auc = self.validate(self.swa_model)
            self.logger.info(
                f"SWA Model Validation | Loss: {swa_val_loss:.15f} | AUC: {swa_val_auc:.15f}"
            )

            # Save SWA model
            swa_save_path = os.path.join(Config.WORKING_DIR, f"{save_name}_swa.pth")
            utils.save_checkpoint(
                {"state_dict": self.swa_model.state_dict(), "auc": swa_val_auc},
                is_best=False,
                filepath=swa_save_path,
            )

            return self.swa_model, swa_val_auc
        else:
            # If we didn't reach SWA or stopped early, return the best base model
            # We need to reload the best weights
            base_save_path = os.path.join(
                Config.WORKING_DIR, f"{save_name}_base_best.pth"
            )
            if os.path.exists(base_save_path):
                checkpoint = torch.load(base_save_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["state_dict"])
                self.logger.info(
                    f"Loaded best base model from epoch {checkpoint['epoch']} with AUC {checkpoint['best_auc']:.15f}"
                )

            return self.model, self.best_auc
