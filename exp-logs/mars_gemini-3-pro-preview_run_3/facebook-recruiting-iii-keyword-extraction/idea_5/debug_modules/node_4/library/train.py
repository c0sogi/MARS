import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import (
    get_logger,
    set_seed,
    AverageMeter,
    calculate_metrics,
    EarlyStopping,
)
from library.model import DualStreamTextCNN
from library.data import get_dataloaders

logger = get_logger("train_module")


class Trainer:
    """
    Manages the training lifecycle of the Dual-Stream TextCNN.
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
        scaler,
        early_stopping,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.early_stopping = early_stopping

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()
        f1_scores = AverageMeter()
        start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            # Move data to device
            title = batch["title"].to(self.device, non_blocking=True)
            body = batch["body"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast():
                logits = self.model(title, body)
                loss = self.criterion(logits, targets)

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()

            # Unscale for gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            # Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler Step
            self.scheduler.step()

            # Metrics
            batch_f1 = calculate_metrics(logits, targets)
            losses.update(loss.item(), title.size(0))
            f1_scores.update(batch_f1, title.size(0))

            # Log periodically (e.g., every 10% of batches or fixed interval)
            if (i + 1) % 100 == 0:
                logger.info(
                    f"Epoch [{epoch}][{i+1}/{len(self.train_loader)}] "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"F1: {f1_scores.val:.4f} ({f1_scores.avg:.4f}) "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
                )

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch} Train Summary: "
            f"Loss: {losses.avg:.4f}, F1: {f1_scores.avg:.4f}, "
            f"Time: {elapsed:.2f}s"
        )
        return losses.avg, f1_scores.avg

    def validate(self, epoch):
        """
        Runs validation phase.
        """
        self.model.eval()
        losses = AverageMeter()
        f1_scores = AverageMeter()
        start_time = time.time()

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                title = batch["title"].to(self.device, non_blocking=True)
                body = batch["body"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)

                # Forward pass (autocast optional for inference but good for consistency)
                with autocast():
                    logits = self.model(title, body)
                    loss = self.criterion(logits, targets)

                batch_f1 = calculate_metrics(logits, targets)
                losses.update(loss.item(), title.size(0))
                f1_scores.update(batch_f1, title.size(0))

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch} Val Summary: "
            f"Loss: {losses.avg}, F1: {f1_scores.avg}, "  # Printing full precision
            f"Time: {elapsed:.2f}s"
        )
        return losses.avg, f1_scores.avg

    def fit(self, num_epochs):
        """
        Main training loop.
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            self.train_epoch(epoch)
            val_loss, val_f1 = self.validate(epoch)

            # Check Early Stopping
            self.early_stopping(val_f1, self.model, self.optimizer, epoch)

            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break


def run_training():
    """
    Sets up the environment, data, model, and triggers training.
    """
    set_seed(Config.SEED)
    device = Config.get_device()
    logger.info(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, _, vocab, mlb = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    num_classes = len(mlb.classes_)
    logger.info(f"Number of classes: {num_classes}")

    # 2. Initialize Model
    model = DualStreamTextCNN(num_classes=num_classes)
    model.to(device)

    # 3. Setup Optimization
    # Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (OneCycleLR)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.NUM_EPOCHS,
        pct_start=0.1,  # 10% warmup
    )

    # Mixed Precision Scaler
    scaler = GradScaler()

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=True, path=Config.MODEL_SAVE_PATH
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        early_stopping=early_stopping,
    )

    # 5. Start Training
    trainer.fit(Config.NUM_EPOCHS)
