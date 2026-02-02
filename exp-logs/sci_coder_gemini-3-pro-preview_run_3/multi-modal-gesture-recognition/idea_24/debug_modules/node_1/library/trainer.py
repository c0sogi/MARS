import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random

from library import config
from library import utils
from library import dataset
from library import model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self):
        # Set reproducibility
        set_seed(config.SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

        # Initialize Datasets and Loaders
        # Using load_cached_data=True as per requirements
        self.train_dataset = dataset.GestureDataset(mode="train", load_cached_data=True)
        self.val_dataset = dataset.GestureDataset(mode="val", load_cached_data=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Initialize Model
        self.model = model.GI_HCSN().to(self.device)

        # Optimization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Configuration
        # Move class weights to device
        self.class_weights = config.CLASS_WEIGHTS.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.smoothing_threshold = config.SMOOTHING_THRESHOLD
        self.smoothing_weight = config.SMOOTHING_WEIGHT

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: returns logits for Stage 1, 2, 3
            logits_1, logits_2, logits_3 = self.model(features)

            # Flatten for CrossEntropy: (Batch * Time, NumClasses) vs (Batch * Time)
            # Reshape logits: (Batch, Time, Classes) -> (Batch * Time, Classes)
            # Reshape labels: (Batch, Time) -> (Batch * Time)

            # --- Cross Entropy Losses ---
            loss_1 = self.criterion(
                logits_1.reshape(-1, config.NUM_CLASSES), labels.reshape(-1)
            )
            loss_2 = self.criterion(
                logits_2.reshape(-1, config.NUM_CLASSES), labels.reshape(-1)
            )
            loss_3 = self.criterion(
                logits_3.reshape(-1, config.NUM_CLASSES), labels.reshape(-1)
            )

            # --- Smoothing Losses (Stage 2 & 3) ---
            # Convert logits to log_probs for truncated_mse_loss
            log_probs_2 = F.log_softmax(logits_2, dim=2)
            log_probs_3 = F.log_softmax(logits_3, dim=2)

            smooth_loss_2 = utils.truncated_mse_loss(
                log_probs_2, threshold=self.smoothing_threshold
            )
            smooth_loss_3 = utils.truncated_mse_loss(
                log_probs_3, threshold=self.smoothing_threshold
            )

            # --- Total Loss ---
            # Cascaded Loss + Smoothing
            total_loss = (
                loss_1
                + loss_2
                + loss_3
                + self.smoothing_weight * (smooth_loss_2 + smooth_loss_3)
            )

            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()

            # Calculate Accuracy on Stage 3 (Final Output)
            # Argmax over class dimension
            preds = torch.argmax(logits_3, dim=2)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.numel()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_preds / total_preds if total_preds > 0 else 0.0

        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits_1, logits_2, logits_3 = self.model(features)

                # Cross Entropy
                loss_1 = self.criterion(
                    logits_1.reshape(-1, config.NUM_CLASSES), labels.reshape(-1)
                )
                loss_2 = self.criterion(
                    logits_2.reshape(-1, config.NUM_CLASSES), labels.reshape(-1)
                )
                loss_3 = self.criterion(
                    logits_3.reshape(-1, config.NUM_CLASSES), labels.reshape(-1)
                )

                # Smoothing
                log_probs_2 = F.log_softmax(logits_2, dim=2)
                log_probs_3 = F.log_softmax(logits_3, dim=2)
                smooth_loss_2 = utils.truncated_mse_loss(
                    log_probs_2, threshold=self.smoothing_threshold
                )
                smooth_loss_3 = utils.truncated_mse_loss(
                    log_probs_3, threshold=self.smoothing_threshold
                )

                total_loss = (
                    loss_1
                    + loss_2
                    + loss_3
                    + self.smoothing_weight * (smooth_loss_2 + smooth_loss_3)
                )

                running_loss += total_loss.item()

                preds = torch.argmax(logits_3, dim=2)
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.numel()

        val_loss = running_loss / len(self.val_loader)
        val_acc = correct_preds / total_preds if total_preds > 0 else 0.0

        return val_loss, val_acc

    def fit(self):
        print(f"Starting training for {config.NUM_EPOCHS} epochs...")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(config.NUM_EPOCHS):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(
                f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss}, Train Acc: {train_acc}, "
                f"Val Loss: {val_loss}, Val Acc: {val_acc}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), config.BEST_MODEL_PATH)
                print(
                    f"New best model saved at epoch {epoch+1} with Val Loss: {val_loss}"
                )
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")
