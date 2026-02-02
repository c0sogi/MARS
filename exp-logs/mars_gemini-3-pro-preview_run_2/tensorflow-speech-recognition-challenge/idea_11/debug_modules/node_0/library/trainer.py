import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import (
    WORKING_DIR,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    INT2LABEL,
    SEED,
    PRETRAINED,
)
from library.utils import set_seed
from library.model import AudioEfficientNet


class Trainer:
    """
    Trainer class to manage the training, validation, and prediction processes
    for the Audio EfficientNet model.
    """

    def __init__(self, train_loader, val_loader, test_loader):
        """
        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            test_loader (DataLoader): DataLoader for the test set.
        """
        # Ensure reproducibility
        set_seed(SEED)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Trainer on device: {self.device}")

        # Initialize Model
        self.model = AudioEfficientNet(pretrained=PRETRAINED)
        self.model.to(self.device)

        # Loss Function
        # Label smoothing helps prevent the model from becoming overconfident
        self.criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        # Optimizer
        # AdamW is generally robust for transformer/CNN architectures
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Learning Rate Scheduler
        # Cosine Annealing reduces LR smoothly to 0 by the end of training
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS
        )

        # Checkpoint path
        self.best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

        # Tracking
        self.best_val_acc = 0.0

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        """Runs evaluation on the validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self, epochs=EPOCHS, patience=5):
        """
        Runs the full training loop with Early Stopping.

        Args:
            epochs (int): Maximum number of epochs to train.
            patience (int): Number of epochs to wait for improvement before stopping.
        """
        print(f"Starting training for {epochs} epochs with patience {patience}...")
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Train and Validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update Scheduler
            self.scheduler.step()

            duration = time.time() - start_time

            # Print metrics with full precision
            print(f"Epoch {epoch + 1}/{epochs} | Time: {duration:.2f}s")
            print(f"  Train Loss: {train_loss} | Train Acc: {train_acc}")
            print(f"  Val Loss:   {val_loss} | Val Acc:   {val_acc}")

            # Checkpointing and Early Stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  -> New best model saved! (Accuracy: {val_acc})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_val_acc}")

    def predict(self):
        """
        Generates predictions for the test set using the best saved model.

        Returns:
            list: A list of predicted string labels.
        """
        # Load the best model weights
        if os.path.exists(self.best_model_path):
            print(f"Loading best model from {self.best_model_path} for inference...")
            state_dict = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print("Warning: Best model file not found. Using current model weights.")

        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for images, _ in self.test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                _, predicted_indices = torch.max(outputs, 1)

                # Move to CPU and convert to numpy
                preds_np = predicted_indices.cpu().numpy()
                all_predictions.extend(preds_np)

        # Map integer indices back to string labels
        # INT2LABEL keys are integers, values are strings
        predicted_labels = [INT2LABEL[idx] for idx in all_predictions]

        return predicted_labels
