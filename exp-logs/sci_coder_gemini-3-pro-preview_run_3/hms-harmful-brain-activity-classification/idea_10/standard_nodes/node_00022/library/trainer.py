import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import AverageMeter, get_logger, kl_divergence_score, seed_everything
from library.dataset import EEGDataset
from library.model import MultiResDualStreamNet


class Trainer:
    """
    Manages the training lifecycle of the Multi-Resolution Dual-Stream Network.
    """

    def __init__(self, config=Config, debug=False):
        self.config = config
        self.debug = debug
        self.device = torch.device(self.config.DEVICE)

        # Setup directories
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)
        self.log_path = os.path.join(self.config.WORKING_DIR, "training.log")
        self.logger = get_logger(self.log_path)

        # Reproducibility
        seed_everything(self.config.SEED)

        # Data Loaders
        self.train_loader, self.val_loader = self._get_dataloaders()

        # Model
        self.model = MultiResDualStreamNet(pretrained=True)
        self.model.to(self.device)

        # Optimization
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config.EPOCHS, eta_min=1e-6
        )

        # Loss Function
        # KLDivLoss expects log-probabilities as input and probabilities as target
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def _get_dataloaders(self):
        """Initializes datasets and dataloaders."""
        subset_size = self.config.DEBUG_SUBSET_SIZE if self.debug else None

        self.logger.info(f"Initializing Train Dataset (Debug={self.debug})...")
        train_dataset = EEGDataset(
            mode="train",
            config=self.config,
            load_cached_data=True,
            subset_size=subset_size,
        )

        self.logger.info(f"Initializing Val Dataset (Debug={self.debug})...")
        val_dataset = EEGDataset(
            mode="val",
            config=self.config,
            load_cached_data=True,
            subset_size=subset_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader

    def mixup_data(self, x_a, x_b, y, alpha=0.4):
        """
        Applies MixUp to inputs and targets.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x_a.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x_a = lam * x_a + (1 - lam) * x_a[index, :]
        mixed_x_b = lam * x_b + (1 - lam) * x_b[index, :]
        y_a, y_b = y, y[index]

        return mixed_x_a, mixed_x_b, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Calculates loss for mixed inputs.
        """
        # pred are log_softmax outputs
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter("Loss", ":.4e")

        start_time = time.time()

        for step, ((x_a, x_b), targets) in enumerate(self.train_loader):
            x_a = x_a.to(self.device, non_blocking=True)
            x_b = x_b.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Apply MixUp
            if self.config.USE_MIXUP and self.config.MIXUP_ALPHA > 0:
                x_a, x_b, targets_a, targets_b, lam = self.mixup_data(
                    x_a, x_b, targets, self.config.MIXUP_ALPHA
                )

                logits = self.model((x_a, x_b))
                log_probs = F.log_softmax(logits, dim=1)

                loss = self.mixup_criterion(
                    self.criterion, log_probs, targets_a, targets_b, lam
                )
            else:
                logits = self.model((x_a, x_b))
                log_probs = F.log_softmax(logits, dim=1)
                loss = self.criterion(log_probs, targets)

            loss.backward()

            if self.config.MAX_GRAD_NORM:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.MAX_GRAD_NORM
                )

            self.optimizer.step()
            losses.update(loss.item(), x_a.size(0))

        elapsed = time.time() - start_time
        self.logger.info(
            f"Epoch {epoch} [Train] Loss: {losses.avg:.6f} | Time: {elapsed:.2f}s"
        )

        return losses.avg

    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter("Loss", ":.4e")

        # Store predictions and targets for metric calculation
        all_preds = []
        all_targets = []

        start_time = time.time()

        with torch.no_grad():
            for step, ((x_a, x_b), targets) in enumerate(self.val_loader):
                x_a = x_a.to(self.device, non_blocking=True)
                x_b = x_b.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = self.model((x_a, x_b))

                # Calculate Loss (KLDiv requires log_softmax)
                log_probs = F.log_softmax(logits, dim=1)
                loss = self.criterion(log_probs, targets)
                losses.update(loss.item(), x_a.size(0))

                # Calculate Metric (Requires probabilities)
                probs = F.softmax(logits, dim=1)

                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Competition Metric
        kl_score = kl_divergence_score(all_targets, all_preds)

        elapsed = time.time() - start_time
        self.logger.info(
            f"Epoch {epoch} [Valid] Loss: {losses.avg:.10f} | KL Score: {kl_score:.10f} | Time: {elapsed:.2f}s"
        )

        return losses.avg, kl_score

    def fit(self):
        self.logger.info(f"Starting training on device: {self.device}")
        best_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.config.WORKING_DIR, "best_model.pth")

        for epoch in range(1, self.config.EPOCHS + 1):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_score = self.validate(epoch)

            # Scheduler Step
            self.scheduler.step()

            # Checkpointing & Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info(
                    f"New best model saved with Val Loss: {best_loss:.10f}"
                )
            else:
                patience_counter += 1
                self.logger.info(
                    f"EarlyStopping counter: {patience_counter} out of {self.config.PATIENCE}"
                )

            if patience_counter >= self.config.PATIENCE:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training complete. Best Val Loss: {best_loss:.10f}")
        return best_loss
