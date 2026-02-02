import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.model import ConvNeXtAudio
from library.dataset import get_dataloaders


class Trainer:
    """
    Trainer class to handle model training, validation, and inference.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Initialize Model
        self.model = ConvNeXtAudio(
            num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
        )
        self.model.to(self.device)

        # Loss Function with Label Smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (Cosine Annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self, debug=Config.DEBUG):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        # Get DataLoaders
        train_loader, val_loader, _ = get_dataloaders(debug=debug, load_cached=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    path=Config.BEST_MODEL_PATH,
                    scheduler=self.scheduler,
                )
                print(f"New best model saved with Val Loss: {val_loss}")
            else:
                patience_counter += 1
                print(
                    f"EarlyStopping counter: {patience_counter} out of {Config.PATIENCE}"
                )
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def predict(self, debug=Config.DEBUG):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("Starting inference...")

        # Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"Best model not found at {Config.BEST_MODEL_PATH}. Run fit() first."
            )

        checkpoint = load_checkpoint(
            Config.BEST_MODEL_PATH, self.model, device=self.device
        )
        print(
            f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}"
        )

        # Get Test Loader
        _, _, test_loader = get_dataloaders(debug=debug, load_cached=True)

        self.model.eval()
        all_preds = []
        all_fnames = test_loader.dataset.fnames

        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(self.device)

                outputs = self.model(features)

                # Get predicted class indices
                _, predicted_ids = torch.max(outputs, 1)
                all_preds.extend(predicted_ids.cpu().numpy())

        # Map IDs to Labels
        predicted_labels = [Config.ID2LABEL[idx] for idx in all_preds]

        # Create Submission DataFrame
        df_submission = pd.DataFrame({"fname": all_fnames, "label": predicted_labels})

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Total predictions: {len(df_submission)}")
