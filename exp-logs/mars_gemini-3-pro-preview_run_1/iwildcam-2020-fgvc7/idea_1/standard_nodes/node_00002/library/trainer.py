import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    DEVICE,
    NUM_CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    NUM_WORKERS,
    MODEL_NAME,
    PRETRAINED,
)
from library.utils import set_seed, calculate_class_weights
from library.dataset import create_datasets
from library.model import DetectorGuidedCNN


class Trainer:
    def __init__(self, load_cached_data=True, max_samples=None):
        """
        Initializes the Trainer with datasets, model, optimizer, and loss function.

        Args:
            load_cached_data (bool): Whether to use cached MegaDetector results.
            max_samples (int, optional): Limit the number of samples for debugging.
        """
        # 1. Set Reproducibility
        set_seed()

        self.device = torch.device(DEVICE)
        print(f"Initializing Trainer on device: {self.device}")

        # 2. Load Datasets
        self.train_dataset, self.val_dataset, self.test_dataset = create_datasets(
            load_cached_data=load_cached_data
        )

        # 3. Debugging: Subset datasets if max_samples is provided
        if max_samples is not None:
            print(f"Debug Mode: Limiting datasets to {max_samples} samples.")
            if len(self.train_dataset.df) > max_samples:
                self.train_dataset.df = self.train_dataset.df.iloc[:max_samples]
            if len(self.val_dataset.df) > max_samples:
                self.val_dataset.df = self.val_dataset.df.iloc[:max_samples]
            # Do not truncate test_dataset; submission requires all rows.

        # 4. Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # 5. Initialize Model
        print(f"Creating model: {MODEL_NAME} (Pretrained: {PRETRAINED})")
        self.model = DetectorGuidedCNN(
            model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED
        )
        self.model.to(self.device)

        # 6. Loss Function with Class Weights
        print("Calculating class weights...")
        weights = calculate_class_weights(self.train_dataset.df, NUM_CLASSES)
        weights = weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # 7. Optimizer and Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Decay LR by 0.1 halfway through epochs
        step_size = max(1, NUM_EPOCHS // 2)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=0.1
        )

        # 8. Training State
        self.best_acc = -1.0
        self.best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    def train_epoch(self, epoch_idx):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        duration = time.time() - start_time

        return epoch_loss, epoch_acc, duration

    def validate(self):
        """Runs validation on the validation set."""
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

        if total == 0:
            return 0.0, 0.0

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self, epochs=NUM_EPOCHS):
        """
        Runs the full training process with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs...")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc, train_time = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Step Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{epochs} | "
                f"LR: {current_lr} | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc} | "
                f"Time: {train_time}s"
            )

            # Early Stopping Check
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(
                    f"Validation accuracy improved. Model saved to {self.best_model_path}"
                )
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the best saved model
        and saves them to the submission file.
        """
        print("Generating submission...")

        # Load best model
        if not os.path.exists(self.best_model_path):
            print("Error: Best model not found. Cannot generate submission.")
            return

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        predictions = []
        ids = []

        with torch.no_grad():
            for images, image_ids in self.test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.cpu().numpy())
                ids.extend(image_ids)

        # Create DataFrame
        submission_df = pd.DataFrame({"Id": ids, "Category": predictions})

        # Save to CSV
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(f"Total predictions: {len(submission_df)}")
