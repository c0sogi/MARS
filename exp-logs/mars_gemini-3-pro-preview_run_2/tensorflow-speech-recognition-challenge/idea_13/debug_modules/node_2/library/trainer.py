import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, Any, Optional

from library.config import PathConfig, ModelConfig, TrainConfig, AudioConfig
from library.dataset import get_dataloaders
from library.model import AudioEfficientNetV2
from library.utils import set_seed, MetricMonitor, ModelEMA


class Trainer:
    """
    Trainer class for the Speech Commands classification task.
    Handles model training, validation, EMA updates, and checkpointing.
    """

    def __init__(self, train_config: Optional[TrainConfig] = None):
        # 1. Load Configurations
        self.path_config = PathConfig()
        self.train_config = train_config if train_config is not None else TrainConfig()
        self.model_config = ModelConfig()
        self.audio_config = AudioConfig()

        # 2. Setup Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 3. Set Seeds for Reproducibility
        set_seed(self.train_config.seed)

        # 4. Initialize DataLoaders
        print("Initializing DataLoaders...")
        self.train_loader, self.val_loader = get_dataloaders(
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            debug=self.train_config.debug,
        )

        # 5. Initialize Model
        print(f"Initializing Model: {self.model_config.model_name}")
        self.model = AudioEfficientNetV2(
            config=self.model_config, num_classes=self.audio_config.num_classes
        )
        self.model.to(self.device)

        # 6. Initialize EMA
        self.ema = None
        if self.model_config.use_ema:
            print(f"Initializing ModelEMA with decay: {self.model_config.ema_decay}")
            self.ema = ModelEMA(
                self.model, decay=self.model_config.ema_decay, device=self.device
            )

        # 7. Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )

        # 8. Initialize Scheduler (Cosine Annealing)
        # Note: Simple CosineAnnealingLR for the duration of epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_config.epochs,
            eta_min=self.train_config.min_lr,
        )

        # 9. Initialize Loss Function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.train_config.label_smoothing
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Runs one epoch of training.
        """
        self.model.train()
        loss_monitor = MetricMonitor()
        acc_monitor = MetricMonitor()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Zero Gradients
            self.optimizer.zero_grad()

            # Forward Pass (Mixed Precision could be added here, keeping it simple float32 for safety/compatibility)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward Pass
            loss.backward()
            self.optimizer.step()

            # Update EMA
            if self.ema:
                self.ema.update(self.model)

            # Calculate Metrics
            batch_size = inputs.size(0)
            loss_monitor.update(loss.item(), batch_size)

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == targets).sum().item()
            accuracy = correct / batch_size
            acc_monitor.update(accuracy, batch_size)

        return {"loss": loss_monitor.result(), "accuracy": acc_monitor.result()}

    def validate_epoch(self) -> Dict[str, float]:
        """
        Runs validation using the EMA model (if available) or the main model.
        """
        # Use EMA model for validation if available, otherwise standard model
        if self.ema:
            eval_model = self.ema.module()
        else:
            eval_model = self.model

        eval_model.eval()
        loss_monitor = MetricMonitor()
        acc_monitor = MetricMonitor()

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = eval_model(inputs)
                loss = self.criterion(outputs, targets)

                batch_size = inputs.size(0)
                loss_monitor.update(loss.item(), batch_size)

                _, predicted = torch.max(outputs, 1)
                correct = (predicted == targets).sum().item()
                accuracy = correct / batch_size
                acc_monitor.update(accuracy, batch_size)

        return {"loss": loss_monitor.result(), "accuracy": acc_monitor.result()}

    def save_checkpoint(self, path: str, is_best: bool = False):
        """
        Saves the model checkpoint.
        If using EMA, we prefer saving the EMA weights as the 'best' model.
        """
        # Determine which state dict to save
        if self.ema:
            state_dict = self.ema.module().state_dict()
        else:
            state_dict = self.model.state_dict()

        checkpoint = {
            "state_dict": state_dict,
            "config": self.model_config,
            "audio_config": self.audio_config,
        }

        torch.save(checkpoint, path)
        if is_best:
            print(f"Saved best model to {path}")

    def train(self):
        """
        Main training loop with Early Stopping.
        """
        print("Starting training...")
        best_acc = 0.0
        patience_counter = 0

        start_time = time.time()

        for epoch in range(1, self.train_config.epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate_epoch()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            epoch_time = time.time() - epoch_start

            # Logging
            print(
                f"Epoch {epoch}/{self.train_config.epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"LR: {current_lr:.8f}"
            )
            print(
                f"  Train Loss: {train_metrics['loss']} | Train Acc: {train_metrics['accuracy']}"
            )
            print(
                f"  Val Loss:   {val_metrics['loss']} | Val Acc:   {val_metrics['accuracy']}"
            )

            # Checkpoint & Early Stopping Logic
            current_acc = val_metrics["accuracy"]

            # Save Last Checkpoint
            self.save_checkpoint(self.path_config.last_checkpoint_path)

            if current_acc > best_acc:
                best_acc = current_acc
                patience_counter = 0
                self.save_checkpoint(
                    self.path_config.model_checkpoint_path, is_best=True
                )
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{self.train_config.early_stopping_patience}"
                )

            if patience_counter >= self.train_config.early_stopping_patience:
                print("Early stopping triggered.")
                break

        total_time = time.time() - start_time
        print(
            f"Training finished in {total_time:.2f}s. Best Validation Accuracy: {best_acc}"
        )
