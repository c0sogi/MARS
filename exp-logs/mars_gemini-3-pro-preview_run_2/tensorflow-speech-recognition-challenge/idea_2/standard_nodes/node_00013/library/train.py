import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import AudioEfficientNet


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_path,
    ):
        """
        Trainer class to handle model training and validation.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.best_acc = 0.0

    def train_one_epoch(self, epoch_index):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

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
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self, num_epochs, patience=5):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            end_time = time.time()
            duration = end_time - start_time

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Time: {duration}s")
            print(f"Train Loss: {train_loss}")
            print(f"Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Acc: {val_acc}")

            # Checkpoint
            if val_acc > self.best_acc:
                print(
                    f"Validation accuracy improved from {self.best_acc} to {val_acc}. Saving model..."
                )
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(
                    f"No improvement in validation accuracy. Patience: {epochs_no_improve}/{patience}"
                )

            # Early Stopping
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc}")


def run_training(
    debug=False,
    load_cached_data=True,
    epochs=Config.NUM_EPOCHS,
    patience=5,
    batch_size=Config.BATCH_SIZE,
):
    """
    Main function to setup and run the training process.
    """
    # 1. Set Seed
    set_seed(Config.SEED)

    # 2. Prepare DataLoaders
    # Note: batch_size is used inside get_dataloaders via Config,
    # but we can override Config.BATCH_SIZE dynamically if needed before calling it.
    # Here we assume Config holds the correct value or we modify it if necessary.
    if batch_size != Config.BATCH_SIZE:
        Config.BATCH_SIZE = batch_size

    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Initialize Model
    device = Config.DEVICE
    model = AudioEfficientNet(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # 4. Define Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # Use Label Smoothing to improve generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 6. Start Training
    trainer.fit(num_epochs=epochs, patience=patience)

    return trainer.best_acc
