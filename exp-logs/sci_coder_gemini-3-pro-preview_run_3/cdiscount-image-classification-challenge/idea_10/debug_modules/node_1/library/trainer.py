import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.model import HierarchicalMLP, mixup_data, mixup_criterion


class Trainer:
    """
    Manages the training lifecycle of the HierarchicalMLP model.
    Handles optimization, scheduling, MixUp augmentation, multi-task loss calculation,
    evaluation, and checkpointing.
    """

    def __init__(self, device=None):
        """
        Initialize the Trainer with model, optimizer, scheduler, and loss function.
        """
        self.device = device if device else torch.device(Config.DEVICE)

        # Initialize Model
        self.model = HierarchicalMLP().to(self.device)

        # Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (ReduceLROnPlateau)
        # verbose argument is deprecated/removed in newer PyTorch versions
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        # Loss Function
        # Using Label Smoothing to handle noisy fine-grained labels
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    def train_epoch(self, loader):
        """
        Runs one epoch of training with Feature-Space MixUp.
        """
        self.model.train()
        running_loss = 0.0
        correct_l3 = 0
        total = 0

        for features, l1, l2, l3 in loader:
            features = features.to(self.device)
            l1 = l1.to(self.device)
            l2 = l2.to(self.device)
            l3 = l3.to(self.device)

            # Apply Feature-Space MixUp
            # Interpolates features and creates pairs of targets for all hierarchy levels
            features, l1_a, l1_b, l2_a, l2_b, l3_a, l3_b, lam = mixup_data(
                features, l1, l2, l3, alpha=Config.MIXUP_ALPHA, device=self.device
            )

            self.optimizer.zero_grad()

            # Forward pass
            p1, p2, p3 = self.model(features)

            # Calculate Multi-Task Loss with MixUp
            # Loss is the sum of CrossEntropy for Level 1, Level 2, and Level 3
            loss_l1 = mixup_criterion(self.criterion, p1, l1_a, l1_b, lam)
            loss_l2 = mixup_criterion(self.criterion, p2, l2_a, l2_b, lam)
            loss_l3 = mixup_criterion(self.criterion, p3, l3_a, l3_b, lam)

            total_loss = loss_l1 + loss_l2 + loss_l3

            # Backpropagation
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item() * features.size(0)

            # Calculate approximate accuracy for the target level (L3)
            # We compare prediction against the dominant label in the mix
            _, predicted = torch.max(p3.data, 1)
            target = l3_a if lam > 0.5 else l3_b
            total += l3.size(0)
            correct_l3 += (predicted == target).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct_l3 / total
        return epoch_loss, epoch_acc

    def evaluate(self, loader):
        """
        Evaluates the model on the validation set without MixUp.
        """
        self.model.eval()
        running_loss = 0.0
        correct_l3 = 0
        total = 0

        with torch.no_grad():
            for features, l1, l2, l3 in loader:
                features = features.to(self.device)
                l1 = l1.to(self.device)
                l2 = l2.to(self.device)
                l3 = l3.to(self.device)

                # Forward pass
                p1, p2, p3 = self.model(features)

                # Standard Multi-Task Loss
                loss_l1 = self.criterion(p1, l1)
                loss_l2 = self.criterion(p2, l2)
                loss_l3 = self.criterion(p3, l3)

                total_loss = loss_l1 + loss_l2 + loss_l3

                running_loss += total_loss.item() * features.size(0)

                # Calculate Accuracy for Target Level (L3)
                _, predicted = torch.max(p3.data, 1)
                total += l3.size(0)
                correct_l3 += (predicted == l3).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct_l3 / total
        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on device: {self.device}")

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            # Train and Validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            # Update Learning Rate Scheduler based on Validation Accuracy
            self.scheduler.step(val_acc)

            duration = time.time() - start_time

            # Print metrics (Full precision as requested)
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Time: {duration}s")
            print(f"Train Loss: {train_loss} | Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss} | Val Acc: {val_acc}")

            # Checkpointing
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT)
                print(f"New best model saved to {Config.MODEL_CHECKPOINT}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            # Early Stopping
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # Load the best model weights before returning
        if os.path.exists(Config.MODEL_CHECKPOINT):
            self.model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT))
            print("Loaded best model weights.")

        return self.model
