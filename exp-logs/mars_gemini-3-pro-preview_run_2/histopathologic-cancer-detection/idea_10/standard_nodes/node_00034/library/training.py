import time
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_roc_auc, save_checkpoint, get_logger
from library.model import get_model, get_loss_fn


class ModelEMA:
    """
    Exponential Moving Average for model parameters.
    Maintains a shadow copy of the model that is updated with a decay factor.
    """

    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        # Ensure EMA parameters do not require gradients
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters.
        """
        with torch.no_grad():
            msd = model.state_dict()
            for name, ema_v in self.ema.state_dict().items():
                if name in msd:
                    model_v = msd[name]
                    if model_v.is_floating_point():
                        ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)
                    else:
                        # Copy non-floating point parameters (e.g., num_batches_tracked)
                        ema_v.copy_(model_v)


class Trainer:
    """
    Trainer class responsible for training and validating a single fold.
    """

    def __init__(self, fold, train_loader, val_loader):
        self.fold = fold
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.device
        self.logger = get_logger(f"fold_{fold}")

        # Initialize Model
        self.model = get_model().to(self.device)

        # Initialize EMA if enabled
        self.ema = ModelEMA(self.model, Config.ema_decay) if Config.use_ema else None

        # Loss Function
        self.criterion = get_loss_fn().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        # Scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.epochs, eta_min=Config.min_lr
        )

        self.best_auc = 0.0

    def mixup_data(self, x, y, alpha=1.0):
        """
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
        losses = AverageMeter()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device).view(-1, 1)

            # Apply Mixup
            images, labels_a, labels_b, lam = self.mixup_data(
                images, labels, Config.mixup_alpha
            )

            # Forward pass
            # Note: Model returns stacked logits (B, MSD, C) in training mode
            outputs = self.model(images)

            # Calculate Loss
            loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)

            # Backward and Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update EMA
            if self.ema:
                self.ema.update(self.model)

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, model_to_validate):
        """
        Runs validation on the full hold-out set.
        """
        model_to_validate.eval()
        losses = AverageMeter()
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).view(-1, 1)

                # Forward pass
                # Note: Model returns averaged logits (B, C) in eval mode
                outputs = model_to_validate(images)

                # Calculate Loss (Standard BCE)
                loss = self.criterion(outputs, labels)
                losses.update(loss.item(), images.size(0))

                # Apply Sigmoid for probabilities
                probs = torch.sigmoid(outputs)

                preds_list.append(probs.cpu())
                targets_list.append(labels.cpu())

        # Concatenate all predictions and targets
        preds = torch.cat(preds_list)
        targets = torch.cat(targets_list)

        # Calculate AUC
        auc = calculate_roc_auc(targets, preds)

        return losses.avg, auc

    def fit(self):
        """
        Main training loop.
        """
        self.logger.info(f"Starting training for Fold {self.fold}")

        for epoch in range(Config.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            # Use EMA model for validation if available, otherwise standard model
            val_model = self.ema.ema if self.ema else self.model
            val_loss, val_auc = self.validate(val_model)

            # Step Scheduler
            self.scheduler.step()

            # Logging
            elapsed = time.time() - start_time
            self.logger.info(
                f"Fold {self.fold} | Epoch {epoch + 1}/{Config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc} | "
                f"Time: {elapsed:.2f}s"
            )

            # Save Checkpoint
            is_best = val_auc > self.best_auc
            if is_best:
                self.best_auc = val_auc

            # Save the EMA model state as the checkpoint
            save_checkpoint(
                val_model.state_dict(),
                is_best=is_best,
                filepath=f"{Config.checkpoints_dir}/last_model_fold_{self.fold}.pth",
                best_filepath=f"{Config.checkpoints_dir}/best_model_fold_{self.fold}.pth",
            )

        self.logger.info(f"Fold {self.fold} finished. Best AUC: {self.best_auc}")
