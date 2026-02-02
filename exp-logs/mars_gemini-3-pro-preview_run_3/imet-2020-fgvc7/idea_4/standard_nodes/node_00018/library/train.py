import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.models import get_model
from library.loss import AsymmetricLoss
from library.utils import calculate_f1_score, optimize_threshold, save_checkpoint


class Trainer:
    """
    Manages the training and validation process for a single model.
    """

    def __init__(self, model, optimizer, scheduler, criterion, device, scaler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.scaler = scaler
        self.best_score = 0.0

    def train_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        for i, (images, targets, _) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / num_batches
        return avg_loss

    def validate(self, val_loader):
        """
        Runs validation and calculates metrics.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Mixed precision inference
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(outputs)

                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = running_loss / len(val_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate F1 Score (Standard 0.5 threshold)
        f1_score = calculate_f1_score(all_targets, all_preds, threshold=0.5)

        # Calculate Optimized F1 Score (for monitoring potential)
        best_thresh, opt_f1 = optimize_threshold(all_targets, all_preds, num_steps=50)

        return avg_loss, f1_score, opt_f1, best_thresh

    def fit(self, train_loader, val_loader, epochs, model_name, patience=5):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training for {model_name}...")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_f1, opt_f1, best_thresh = self.validate(val_loader)

            elapsed = time.time() - start_time

            # Print full precision metrics
            print(f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s")
            print(f"  Train Loss: {train_loss}")
            print(f"  Val Loss:   {val_loss}")
            print(f"  Val F1:     {val_f1}")
            print(f"  Opt F1:     {opt_f1} (Thresh: {best_thresh})")

            # Checkpointing
            # We use the standard F1 (0.5) or Optimized F1?
            # Usually for multi-label with unknown distribution shifts, optimizing threshold is safer.
            # However, for consistency with the prompt's metric definition, we track improvement on Opt F1
            # as we will calibrate threshold in inference.
            current_score = opt_f1

            if current_score > self.best_score:
                print(
                    f"  Score improved from {self.best_score} to {current_score}. Saving checkpoint."
                )
                self.best_score = current_score
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.best_score,
                    f"{model_name}_best.pth",
                )
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        return self.best_score


def train_specific_model(model_name, epochs=Config.EPOCHS, debug=Config.DEBUG):
    """
    Sets up the environment and trains a specific model architecture.

    Args:
        model_name (str): Name of the model to train (e.g., 'resnet101d').
        epochs (int): Number of epochs to train.
        debug (bool): Whether to run in debug mode (subset of data).
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing {model_name} on {device}...")

    # 2. Data
    # load_cached_data=True ensures we use the parquet cache if available
    train_loader, val_loader, _ = get_dataloaders(
        debug=debug,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model
    model = get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR needs total steps
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # Warmup for first 10%
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 5. Loss & Scaler
    criterion = AsymmetricLoss(
        gamma_neg=Config.ASL_GAMMA_NEG,
        gamma_pos=Config.ASL_GAMMA_POS,
        clip=Config.ASL_CLIP,
    )
    scaler = GradScaler()

    # 6. Trainer
    trainer = Trainer(model, optimizer, scheduler, criterion, device, scaler)

    # 7. Execute
    best_score = trainer.fit(train_loader, val_loader, epochs, model_name)

    print(f"Training finished for {model_name}. Best F1 Score: {best_score}")

    # Clear memory
    del model, optimizer, scheduler, scaler, trainer, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_score
