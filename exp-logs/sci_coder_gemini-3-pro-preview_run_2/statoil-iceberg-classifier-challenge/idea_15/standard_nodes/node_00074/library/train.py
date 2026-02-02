import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, model, data_loader


class Trainer:
    """
    Manages the training, validation, and optimization of the GDP-Net model.
    """

    def __init__(
        self, model, device, criterion, optimizer, scheduler=None, patience=10
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_loss = float("inf")

    def train_one_epoch(self, train_loader, epoch_idx):
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for batch_idx, (images, angles, labels) in enumerate(train_loader):
            images = images.to(self.device)
            angles = angles.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # Shape: [Batch, 1]

            self.optimizer.zero_grad()

            # Forward pass: pass both image and incidence angle
            outputs = self.model(images, angles)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            preds = torch.sigmoid(outputs) > 0.5
            correct_preds += torch.sum(preds == (labels > 0.5)).item()
            total_preds += images.size(0)

        epoch_loss = running_loss / total_preds
        epoch_acc = correct_preds / total_preds

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(self.device)
                angles = angles.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                outputs = self.model(images, angles)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs) > 0.5
                correct_preds += torch.sum(preds == (labels > 0.5)).item()
                total_preds += images.size(0)

        epoch_loss = running_loss / total_preds
        epoch_acc = correct_preds / total_preds

        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader, max_epochs):
        print(f"Starting training on device: {self.device}")

        early_stopping_counter = 0

        for epoch in range(max_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)

            duration = time.time() - start_time

            # Print metrics with full precision
            print(f"Epoch {epoch+1}/{max_epochs} | Time: {duration:.2f}s")
            print(f"  Train Loss: {train_loss:.10f} | Train Acc: {train_acc:.6f}")
            print(f"  Val Loss:   {val_loss:.10f} | Val Acc:   {val_acc:.6f}")

            # Scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Early Stopping Logic
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
                # print("  -> Validation loss improved. Saving model state.")
            else:
                early_stopping_counter += 1
                # print(f"  -> No improvement. Counter: {early_stopping_counter}/{self.patience}")

            if early_stopping_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Validation Loss: {self.best_loss:.10f}")

        # Load best weights
        self.model.load_state_dict(self.best_model_wts)
        return self.model


def train_fold(fold_index):
    """
    Trains a single fold of the GDP-Net model.
    """
    print(f"\n{'='*20} Training Fold {fold_index} {'='*20}")

    # 1. Data Loading
    train_loader, val_loader = data_loader.get_fold_loaders(
        fold_index, load_cached_data=True
    )

    # 2. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.DPCNet().to(device)

    # 3. Setup Training Components
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=net,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=config.PATIENCE,
    )

    # 5. Execute Training
    best_model = trainer.fit(train_loader, val_loader, max_epochs=config.MAX_EPOCHS)

    # 6. Save Model
    save_path = os.path.join(config.WORKING_DIR, f"gdpnet_fold_{fold_index}.pth")
    utils.save_checkpoint(best_model, save_path)
    print(f"Model for fold {fold_index} saved to {save_path}")


def train_all_folds():
    """
    Orchestrates the training of all folds defined in config.
    """
    utils.seed_everything(config.SEED)

    for fold in range(config.NUM_FOLDS):
        train_fold(fold)
