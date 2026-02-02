import time
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_macro_f1, save_checkpoint


class Trainer:
    """
    Manages the training and validation process for the Herbarium Plant Species Classification model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device=Config.DEVICE,
        patience=Config.PATIENCE,
    ):
        """
        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): The optimizer.
            scheduler (LRScheduler): The learning rate scheduler.
            device (str): Device to run training on ('cuda' or 'cpu').
            patience (int): Number of epochs to wait for improvement before early stopping.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience

        # Cite solution_lesson_node_00003: Label Smoothing (0.1) to improve convergence with massive classification heads
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # Initialize GradScaler for Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler("cuda") if self.device == "cuda" else None

    def train_one_epoch(self, epoch_index):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        start_time = time.time()

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Training
            if self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_loss = running_loss / total_samples
        duration = time.time() - start_time

        print(
            f"Epoch [{epoch_index+1}] Training Loss: {avg_loss} (Time: {duration:.2f}s)"
        )
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set and calculates metrics.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Use mixed precision for inference as well to speed it up
                if self.device == "cuda":
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # Get predictions
                _, preds = torch.max(outputs, 1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = running_loss / total_samples

        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Calculate Macro F1
        macro_f1 = calculate_macro_f1(all_labels, all_preds)

        print(f"Validation Loss: {avg_loss}")
        print(f"Validation Macro F1: {macro_f1}")

        return avg_loss, macro_f1

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        """
        Main training loop with early stopping.
        """
        best_f1 = 0.0
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs on {self.device}...")

        for epoch in range(num_epochs):
            # Train
            self.train_one_epoch(epoch)

            # Validate
            val_loss, val_f1 = self.validate()

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            # Checkpoint and Early Stopping
            if val_f1 > best_f1:
                print(f"New best model found! F1 improved from {best_f1} to {val_f1}")
                best_f1 = val_f1
                patience_counter = 0

                # Save best model
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_f1": best_f1,
                }
                save_checkpoint(state, is_best=True, filename=Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{self.patience}")

            if patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation F1: {best_f1}")
