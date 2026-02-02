import os
import time
import numpy as np
import torch
import torch.nn as nn
from library.config import train_config, path_config
from library.utils import set_seed
from library.transforms import get_augmentations


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_config.epochs, eta_min=train_config.eta_min
        )

        # Augmentations (SpecAugment)
        self.augmentations = get_augmentations().to(self.device)

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 1. Apply SpecAugment
            with torch.no_grad():
                inputs = self.augmentations(inputs)

            # 2. Apply Mixup
            alpha = train_config.mixup_alpha
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.0

            batch_size = inputs.size(0)
            index = torch.randperm(batch_size).to(self.device)

            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            targets_a, targets_b = targets, targets[index]

            # 3. Forward Pass
            self.optimizer.zero_grad()
            outputs = self.model(mixed_inputs)

            # 4. Mixup Loss
            loss = lam * self.criterion(outputs, targets_a) + (
                1 - lam
            ) * self.criterion(outputs, targets_b)

            # 5. Backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            total += batch_size

            # Estimate accuracy for logging (using the dominant label)
            _, predicted = outputs.max(1)
            # We count it correct if it matches the label with higher weight
            if lam >= 0.5:
                correct += predicted.eq(targets_a).sum().item()
            else:
                correct += predicted.eq(targets_b).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self):
        best_val_acc = 0.0
        patience_counter = 0

        print(f"Starting training on {self.device} for {train_config.epochs} epochs.")

        for epoch in range(train_config.epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            end_time = time.time()
            epoch_duration = end_time - start_time

            print(
                f"Epoch {epoch+1}/{train_config.epochs} | "
                f"Time: {epoch_duration:.2f}s | "
                f"LR: {current_lr} | "
                f"Train Loss: {train_loss} | "
                f"Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | "
                f"Val Acc: {val_acc}"
            )

            # Checkpoint & Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), path_config.model_save_path)
                print(f"New best model saved with Val Acc: {best_val_acc}")
            else:
                patience_counter += 1
                if patience_counter >= train_config.early_stopping_patience:
                    print(
                        f"Early stopping triggered after {patience_counter} epochs without improvement."
                    )
                    break

        print(f"Training complete. Best Validation Accuracy: {best_val_acc}")
