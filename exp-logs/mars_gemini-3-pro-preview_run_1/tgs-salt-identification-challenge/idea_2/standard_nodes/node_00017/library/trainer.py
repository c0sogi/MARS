import os
import torch
import numpy as np
from library.utils import do_kaggle_metric
from library.dataset import ORIG_SIZE, TARGET_SIZE


class ModelTrainer:
    """
    Encapsulates the training logic for the salt segmentation task.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        scheduler=None,
        num_epochs=50,
        patience=10,
        checkpoint_dir="./checkpoints",
        checkpoint_name="best_model.pth",
    ):
        """
        Args:
            model: PyTorch model.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: PyTorch optimizer.
            criterion: Loss function.
            device: torch.device.
            scheduler: Learning rate scheduler (ReduceLROnPlateau).
            num_epochs: Maximum number of epochs.
            patience: Early stopping patience.
            checkpoint_dir: Directory to save checkpoints.
            checkpoint_name: Name of the best model file.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0

        for inputs, masks in self.train_loader:
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_masks = []

        # Calculate padding for cropping back to original size
        # Image was padded from 101x101 to 128x128
        pad_h = TARGET_SIZE - ORIG_SIZE
        pad_top = pad_h // 2
        pad_w = TARGET_SIZE - ORIG_SIZE
        pad_left = pad_w // 2

        with torch.no_grad():
            for inputs, masks in self.val_loader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item() * inputs.size(0)

                # Apply sigmoid to get probabilities
                preds_prob = torch.sigmoid(outputs)

                # Move to CPU and store
                all_preds.append(preds_prob.cpu().numpy())
                all_masks.append(masks.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)

        # Concatenate batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)

        # Center crop predictions and masks back to original size (101x101)
        # Slicing handles both (N, C, H, W) and (N, H, W)
        preds_cropped = all_preds[
            ..., pad_top : pad_top + ORIG_SIZE, pad_left : pad_left + ORIG_SIZE
        ]
        masks_cropped = all_masks[
            ..., pad_top : pad_top + ORIG_SIZE, pad_left : pad_left + ORIG_SIZE
        ]

        # Calculate Kaggle Metric (mAP at IoU thresholds)
        # The metric function handles thresholding of probabilities
        score = do_kaggle_metric(preds_cropped, masks_cropped, threshold=0.5)

        return epoch_loss, score

    def run(self):
        """
        Runs the full training loop with early stopping and scheduling.
        """
        best_score = -float("inf")
        patience_counter = 0

        print("Starting training...")

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_score = self.validate()

            # Step the scheduler if provided. Using val_score (Metric-Aligned Model Selection)
            if self.scheduler:
                self.scheduler.step(val_score)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Score: {val_score}"
            )

            # Checkpoint and Early Stopping Logic based on Metric (Cite Lesson 00016)
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training complete. Best Val Score: {best_score}")
        return best_score
