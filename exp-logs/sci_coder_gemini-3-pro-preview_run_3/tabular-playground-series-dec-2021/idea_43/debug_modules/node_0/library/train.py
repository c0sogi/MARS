import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, get_device
from library.model import DualViewDCNResNet
from library.data_loader import get_dataloaders


class Trainer:
    """
    Manages the training lifecycle of the Deeply-Supervised Dual-View Network.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True,
        )

        # State tracking
        self.best_acc = 0.0
        self.best_model_state = None
        self.early_stop_counter = 0

    def get_annealing_lambda(self, epoch):
        """
        Calculates the auxiliary loss weight lambda_t for the current epoch.
        Linearly decays from ANNEAL_START to ANNEAL_END.
        """
        total_epochs = Config.EPOCHS
        start = Config.ANNEAL_START
        end = Config.ANNEAL_END

        # Avoid division by zero if epochs=1
        if total_epochs <= 1:
            return start

        # Linear interpolation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        # x is epoch, x1=0, x2=total_epochs-1
        progress = epoch / (total_epochs - 1)
        lambda_t = start + progress * (end - start)
        return max(0.0, lambda_t)

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Calculate lambda for this epoch
        lambda_t = self.get_annealing_lambda(epoch_idx)

        for batch_X, batch_y in self.train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits, aux_logits = self.model(batch_X)

            # Primary Loss
            loss_primary = self.criterion(logits, batch_y)

            # Auxiliary Loss (if head is active and lambda > 0)
            loss_aux = 0.0
            if aux_logits is not None and lambda_t > 0:
                loss_aux = self.criterion(aux_logits, batch_y)

            # Combined Annealed Loss
            loss = loss_primary + lambda_t * loss_aux

            # Backward
            loss.backward()
            self.optimizer.step()

            # Metrics
            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(logits, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, lambda_t

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass (Aux head ignored for validation metric)
                logits, _ = self.model(batch_X)

                loss = self.criterion(logits, batch_y)

                running_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self):
        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss, train_acc, lambda_t = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            end_time = time.time()
            duration = end_time - start_time

            # Print metrics
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Time: {duration:.2f}s | "
                f"Lambda: {lambda_t:.4f} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Train Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Acc: {val_acc:.10f}"
            )

            # Scheduler Step
            self.scheduler.step(val_acc)

            # Early Stopping Check
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.early_stop_counter = 0
                # print(f"New best model found! Acc: {self.best_acc:.10f}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc:.10f}")

        # Save best model
        if self.best_model_state is not None:
            print(f"Saving best model to {Config.BEST_MODEL_PATH}")
            torch.save(self.best_model_state, Config.BEST_MODEL_PATH)
        else:
            print("Warning: No best model state found (training might have failed).")


def run_training():
    # 1. Setup
    seed_everything(Config.SEED, deterministic=Config.DETERMINISTIC_CUDNN)
    device = get_device()

    # 2. Data
    # load_cached_data=True allows skipping re-processing if files exist
    train_loader, val_loader, _, input_dim = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = DualViewDCNResNet(input_dim=input_dim, num_classes=Config.NUM_CLASSES)
    model.to(device)

    # 4. Train
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()


if __name__ == "__main__":
    run_training()
