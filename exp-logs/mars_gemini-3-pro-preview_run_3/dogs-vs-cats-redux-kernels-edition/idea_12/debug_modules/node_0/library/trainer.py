import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import get_device


class Trainer:
    """
    Trainer class to handle training, validation, and inference for the Dog vs Cat classification task.
    Encapsulates the training loop, mixed-precision handling, and model checkpointing.
    """

    def __init__(self, model, optimizer, scheduler=None, device=None, save_path=None):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            device (torch.device, optional): Device to run training on.
            save_path (str, optional): Path to save the best model checkpoint.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else get_device()
        self.save_path = (
            save_path
            if save_path
            else os.path.join(Config.WORKING_DIR, "model_best.pth")
        )

        # Loss function: BCEWithLogitsLoss without label smoothing as per strategy
        # This allows the Multi-Sample Dropout head to optimize for extreme probabilities
        self.criterion = nn.BCEWithLogitsLoss()

        # Mixed precision scaler for A100 optimization
        self.scaler = GradScaler()

        self.best_val_loss = float("inf")
        self.model.to(self.device)

    def train_one_epoch(self, train_loader, epoch):
        """
        Trains the model for one epoch using Mixed Precision.

        Args:
            train_loader (DataLoader): The training data loader.
            epoch (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            # Ensure labels are (B, 1) to match model output
            labels = labels.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass with scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate loss
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (DataLoader): The validation data loader.

        Returns:
            tuple: (average_loss, predictions, true_labels)
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply sigmoid to get probabilities
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss = running_loss / dataset_size
        return val_loss, np.concatenate(all_preds), np.concatenate(all_labels)

    def fit(self, train_loader, val_loader, epochs, patience=3):
        """
        Runs the full training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            patience (int): Epochs to wait for improvement before stopping.
        """
        print(f"Starting training on {self.device} for {epochs} epochs.")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, _, _ = self.evaluate(val_loader)

            # Step scheduler if it exists (CosineAnnealingLR steps per epoch)
            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start_time

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{epochs} - Time: {elapsed:.2f}s - "
                f"Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Checkpoint and Early Stopping Logic
            if val_loss < self.best_val_loss:
                print(
                    f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model..."
                )
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        # Load best model for future use/inference
        if os.path.exists(self.save_path):
            print(f"Loading best model from {self.save_path}")
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=self.device)
            )

    def predict(self, test_loader, use_tta=True):
        """
        Generates predictions for the test set.
        Implements Test Time Augmentation (Horizontal Flip) if enabled.

        Args:
            test_loader (DataLoader): Test data.
            use_tta (bool): Whether to use Test Time Augmentation.

        Returns:
            tuple: (probabilities, ids)
        """
        self.model.eval()
        all_preds = []
        all_ids = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device)

                with autocast():
                    # Forward pass original
                    outputs = self.model(images)
                    probs = torch.sigmoid(outputs)

                    if use_tta:
                        # Forward pass flipped (Horizontal Flip)
                        # Images are (B, C, H, W), flip on dim 3 (Width)
                        images_flipped = torch.flip(images, dims=[3])
                        outputs_flipped = self.model(images_flipped)
                        probs_flipped = torch.sigmoid(outputs_flipped)

                        # Average probabilities to stabilize predictions
                        probs = (probs + probs_flipped) / 2.0

                all_preds.append(probs.cpu().numpy())
                all_ids.append(ids.numpy())

        return np.concatenate(all_preds), np.concatenate(all_ids)

    def generate_submission(self, test_loader, output_path=None):
        """
        Generates the submission CSV file using the trained model.

        Args:
            test_loader (DataLoader): Test data.
            output_path (str, optional): Path to save the CSV.
        """
        if output_path is None:
            output_path = Config.SUBMISSION_PATH

        print("Generating predictions for submission...")
        probs, ids = self.predict(test_loader, use_tta=True)

        # Flatten probs (B, 1) -> (B,)
        probs = probs.flatten()
        ids = ids.flatten()

        # Create DataFrame
        df = pd.DataFrame({"id": ids, "label": probs})

        # Ensure ID is int and sort
        df["id"] = df["id"].astype(int)
        df = df.sort_values("id")

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
