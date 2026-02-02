import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, init_logger
from library.model import MultiResResNetCRNN
from library.dataset import get_dataloaders


class Trainer:
    def __init__(self):
        """
        Initializes the Trainer with model, optimizer, criterion, and device.
        """
        self.logger = init_logger()
        self.device = Config.DEVICE

        # Initialize Model
        self.logger.info(f"Initializing model: {Config.MODEL_NAME}")
        self.model = MultiResResNetCRNN().to(self.device)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Best metric tracking
        self.best_val_acc = -1.0

    def train_one_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, dataloader):
        """
        Runs evaluation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, epochs=Config.EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        set_seed(Config.SEED)

        # Get DataLoaders
        train_loader, val_loader, _ = get_dataloaders()

        self.logger.info("Starting training...")

        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Print metrics (Full precision as requested)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Acc: {val_acc}")

            # Checkpoint and Early Stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                self.logger.info(
                    f"New best model found! Saving to {Config.MODEL_SAVE_PATH}"
                )
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(
            f"Training complete. Best Validation Accuracy: {self.best_val_acc}"
        )

    def predict(self):
        """
        Generates predictions for the test set using the best model.
        """
        self.logger.info("Starting prediction on test set...")

        # Load best model
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
            self.logger.info(f"Loaded model from {Config.MODEL_SAVE_PATH}")
        else:
            self.logger.warning("No saved model found. Using current model state.")

        self.model.eval()

        _, _, test_loader = get_dataloaders()

        predictions = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())

        # Map IDs to Labels
        predicted_labels = [Config.ID2LABEL[idx] for idx in predictions]

        # Get Filenames
        # We access the dataframe directly from the dataset to ensure alignment
        # test_loader is sequential (shuffle=False)
        test_df = test_loader.dataset.df

        # Extract basename from filepath (e.g., 'test/audio/clip_000.wav' -> 'clip_000.wav')
        fnames = test_df["filepath"].apply(os.path.basename).tolist()

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"fname": fnames, "label": predicted_labels})

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())
