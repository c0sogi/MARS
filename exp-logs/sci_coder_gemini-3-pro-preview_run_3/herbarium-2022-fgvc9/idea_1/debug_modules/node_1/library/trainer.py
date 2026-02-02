import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import compute_metrics, set_seed
from library.model import PlantClassifier


class Trainer:
    """
    Trainer class to manage the training, validation, and prediction lifecycle
    of the Plant Classifier model.
    """

    def __init__(self, device=None):
        """
        Initialize the Trainer.

        Args:
            device (str, optional): Device to run on ('cuda' or 'cpu').
                                    Defaults to Config.DEVICE.
        """
        # Ensure reproducibility
        set_seed(Config.SEED)

        self.device = device if device else Config.DEVICE

        # Initialize Model
        self.model = PlantClassifier(num_classes=Config.NUM_CLASSES)
        self.model.to(self.device)

        # Optimization components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Paths
        self.best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def train_one_epoch(self, dataloader, scheduler):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if scheduler:
                scheduler.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss

    def validate(self, dataloader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.size(0)

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            val_f1 = compute_metrics(all_labels, all_preds)
        else:
            val_f1 = 0.0

        return epoch_loss, val_f1

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        load_cached_model=False,
    ):
        """
        Runs the full training loop with Early Stopping and Caching.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs (int): Number of epochs to train.
            load_cached_model (bool): If True, attempts to load a pre-trained model
                                      from cache instead of training.
        """
        # Caching Logic
        if load_cached_model:
            if os.path.exists(self.best_model_path):
                print(f"Loading cached model from {self.best_model_path}")
                self.model.load_state_dict(
                    torch.load(self.best_model_path, map_location=self.device)
                )
                return
            else:
                print(
                    f"Cached model not found at {self.best_model_path}. Starting training..."
                )

        # Scheduler Setup (OneCycleLR)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )

        best_f1 = 0.0
        patience = 3
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader, scheduler)
            val_loss, val_f1 = self.validate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1}"
            )

            # Checkpointing and Early Stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Load best model weights before finishing
        if os.path.exists(self.best_model_path):
            print(f"Loading best model weights with F1: {best_f1}")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set and saves them to a CSV file.

        Args:
            test_loader: DataLoader for test data.
        """
        self.model.eval()
        predictions = []
        image_ids = []

        print("Generating predictions for submission...")

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                predictions.extend(preds)
                image_ids.extend(ids)

        # Create submission DataFrame
        df = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
