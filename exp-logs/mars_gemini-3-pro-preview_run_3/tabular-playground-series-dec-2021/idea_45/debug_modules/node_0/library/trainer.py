import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config


class Trainer:
    """
    Handles the training, validation, and inference processes for the Asymmetric Parallel Vector-DCN-ResNet.
    Implements AdamW optimization, ReduceLROnPlateau scheduling, and Early Stopping.
    """

    def __init__(self, model):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
        """
        self.device = torch.device(Config.DEVICE)
        self.model = model.to(self.device)

        # Loss function for multi-class classification
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: AdamW (Decoupled Weight Decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: ReduceLROnPlateau
        # Monitors validation accuracy ('max') and decays LR by factor 0.1
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=Config.SCHEDULER_MODE,
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True,  # Print when LR changes
        )

    def train_one_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        return epoch_loss

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average loss and accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = running_loss / total_samples
        accuracy = correct / total_samples
        return avg_loss, accuracy

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """
        Runs the full training loop with Early Stopping.
        """
        print(f"Starting training on {self.device} for {epochs} epochs...")

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        patience_counter = 0

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.evaluate(val_loader)

            # Scheduler Step
            # ReduceLROnPlateau expects the metric to monitor (Accuracy in 'max' mode)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_acc)

            epoch_duration = time.time() - epoch_start

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Time: {epoch_duration:.2f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Acc: {val_acc}"
            )

            # Early Stopping Logic
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                # print(f"  -> New Best Accuracy! Saving model state.")
            else:
                patience_counter += 1
                # print(f"  -> No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        total_time = time.time() - start_time
        print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"Best Validation Accuracy: {best_acc}")

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return best_acc

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the trained model.
        Returns a numpy array of predicted class indices.
        """
        self.model.eval()
        predictions = []

        print("Generating predictions...")
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)
