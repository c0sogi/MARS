import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import TrainConfig, ModelConfig
from library.utils import (
    set_seed,
    mixup_data,
    mixup_criterion,
    AverageMeter,
    LabelMapper,
)
from library.dataset import get_dataloaders
from library.model import ContextAwareEfficientNet


class Trainer:
    def __init__(self):
        # 1. Setup
        self.config = TrainConfig
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.seed)

        # Ensure working directory exists
        os.makedirs(self.config.work_dir, exist_ok=True)

        # 2. Data
        print("Initializing DataLoaders...")
        self.train_loader, self.val_loader, _ = get_dataloaders(load_cached_data=True)

        # 3. Model
        print("Initializing Model...")
        self.model = ContextAwareEfficientNet().to(self.device)

        # 4. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.T_max,
            eta_min=self.config.eta_min,
        )

        # 5. Loss & Metrics
        self.criterion = nn.CrossEntropyLoss()
        self.mapper = LabelMapper()

        # State
        self.best_acc = 0.0
        self.start_epoch = 0

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Mixup
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, self.config.mixup_alpha, self.device
            )

            # Forward
            outputs = self.model(inputs)
            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), inputs.size(0))

        return losses.avg

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward
                outputs = self.model(inputs)

                # Get predictions (indices 0-30)
                _, pred_indices = torch.max(outputs, 1)

                # Map to submission labels for metric calculation
                # 1. Decode indices to fine-grained strings
                pred_labels_fine = self.mapper.decode(pred_indices)
                target_labels_fine = self.mapper.decode(targets)

                # 2. Map fine-grained strings to 12-class submission strings
                pred_labels_sub = [
                    self.mapper.map_to_submission(l) for l in pred_labels_fine
                ]
                target_labels_sub = [
                    self.mapper.map_to_submission(l) for l in target_labels_fine
                ]

                # 3. Calculate Accuracy
                for p, t in zip(pred_labels_sub, target_labels_sub):
                    if p == t:
                        correct += 1
                    total += 1

        acc = 100.0 * correct / total if total > 0 else 0.0
        return acc

    def fit(self):
        print(f"Starting training on {self.device} for {self.config.epochs} epochs.")

        patience = 10
        patience_counter = 0

        for epoch in range(self.start_epoch, self.config.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_acc = self.validate()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            duration = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Time: {duration:.2f}s | "
                f"LR: {current_lr} | "
                f"Train Loss: {train_loss} | "
                f"Val Acc (12-class): {val_acc}"
            )

            # Checkpoint
            if val_acc > self.best_acc:
                print(
                    f"Validation accuracy improved from {self.best_acc} to {val_acc}. Saving model..."
                )
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.config.checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs of no improvement."
                )
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc}")


def train_model():
    trainer = Trainer()
    trainer.fit()
