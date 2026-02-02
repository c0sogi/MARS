import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import AverageMeter


class ModelTrainer:
    """
    Manages the training, validation, and prediction processes for the tumor detection model.
    """

    def __init__(self, model: nn.Module, device: str = Config.DEVICE):
        """
        Args:
            model (nn.Module): The neural network to train.
            device (str): Computation device ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.device = device

        # Loss function for binary classification (combines Sigmoid + BCELoss)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

    def train_epoch(self, loader) -> float:
        """
        Executes one training epoch.

        Args:
            loader (DataLoader): The training data loader.

        Returns:
            float: The average loss for the epoch.
        """
        self.model.train()
        losses = AverageMeter()

        for images, labels, _ in loader:
            images = images.to(self.device)
            # Reshape labels to (B, 1) to match model output dimensions
            labels = labels.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        # Step the scheduler at the end of the epoch
        self.scheduler.step()

        return losses.avg

    def validate(self, loader):
        """
        Evaluates the model on the validation set.

        Args:
            loader (DataLoader): The validation data loader.

        Returns:
            tuple: (average_loss, auc_score)
        """
        self.model.eval()
        losses = AverageMeter()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                losses.update(loss.item(), images.size(0))

                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(outputs)

                all_labels.append(labels.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        # Concatenate results from all batches
        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)

            # Calculate ROC AUC
            # Handle edge case where batch might only have one class
            if len(np.unique(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_preds)
            else:
                auc = 0.5
        else:
            auc = 0.0

        return losses.avg, auc

    def fit(
        self, train_loader, val_loader, epochs: int = Config.EPOCHS, patience: int = 5
    ):
        """
        Runs the full training loop with early stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            patience (int): Epochs to wait for improvement before stopping.
        """
        best_auc = 0.0
        patience_counter = 0

        print(f"Starting training on {self.device} for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Checkpoint and Early Stopping Logic
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
                print(
                    f"Validation AUC improved. Saved model to {Config.CHECKPOINT_PATH}"
                )
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best AUC: {best_auc}")

        # Reload the best model weights for consistency
        if os.path.exists(Config.CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(Config.CHECKPOINT_PATH, map_location=self.device)
            )

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves them to a CSV file.

        Args:
            test_loader (DataLoader): Test data loader.
        """
        self.model.eval()

        ids_list = []
        probs_list = []

        print("Generating predictions for test set...")

        with torch.no_grad():
            for images, _, ids in test_loader:
                images = images.to(self.device)

                # Test Time Augmentation (TTA) - Cite {solution_lesson_node_00002}
                # Averaging predictions across 4 views: Original, HFlip, VFlip, H+VFlip
                logits1 = self.model(images)
                logits2 = self.model(torch.flip(images, dims=[3]))
                logits3 = self.model(torch.flip(images, dims=[2]))
                logits4 = self.model(torch.flip(images, dims=[2, 3]))

                probs1 = torch.sigmoid(logits1)
                probs2 = torch.sigmoid(logits2)
                probs3 = torch.sigmoid(logits3)
                probs4 = torch.sigmoid(logits4)

                probs = (probs1 + probs2 + probs3 + probs4) / 4.0

                # Flatten to 1D array
                probs_np = probs.cpu().numpy().flatten()

                ids_list.extend(ids)
                probs_list.extend(probs_np)

        # Construct submission DataFrame
        submission_df = pd.DataFrame({"id": ids_list, "label": probs_list})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Total predictions generated: {len(submission_df)}")
