import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import set_seed, calculate_accuracy
from library.dataset import get_dataloaders
from library.model import TimeResolvedEfficientNet


class Trainer:
    """
    Trainer class to manage training, validation, and inference for the Speech Command model.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Initialize Model
        self.model = TimeResolvedEfficientNet().to(self.device)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Checkpoint Path
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimizer step
            self.optimizer.step()

            # Metrics
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += calculate_accuracy(outputs, labels) * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_acc = running_acc / total_samples

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                running_acc += calculate_accuracy(outputs, labels) * batch_size
                total_samples += batch_size

        val_loss = running_loss / total_samples
        val_acc = running_acc / total_samples

        return val_loss, val_acc

    def fit(self, load_cached_data=True):
        """
        Main training loop with Early Stopping.
        """
        print("Initializing Training...")
        train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()

            # Print Metrics (Full Precision)
            print(f"Train Loss: {train_loss}")
            print(f"Train Acc: {train_acc}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Acc: {val_acc}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']}")

            # Early Stopping and Checkpointing
            if val_acc > best_val_acc:
                print(
                    f"Validation Accuracy improved from {best_val_acc} to {val_acc}. Saving model..."
                )
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training Complete. Best Validation Accuracy: {best_val_acc}")

        # Load best weights
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )

    def predict_and_submit(self, load_cached_data=True):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Starting Inference...")

        # Ensure model has best weights loaded
        if os.path.exists(self.best_model_path):
            print(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()

        # Get Test Loader
        _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

        all_preds = []

        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(self.device)

                outputs = self.model(features)

                # Get predicted class indices
                _, preds = torch.max(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())

        # Map indices to labels
        predicted_labels = [Config.ID2LABEL[idx] for idx in all_preds]

        # Load Test Metadata to get filenames
        # The test loader preserves order because shuffle=False and it iterates over the metadata CSV rows
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)

        if len(df_test) != len(predicted_labels):
            raise ValueError(
                f"Mismatch between test set size ({len(df_test)}) and predictions ({len(predicted_labels)})"
            )

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"fname": df_test["fname"], "label": predicted_labels}
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
