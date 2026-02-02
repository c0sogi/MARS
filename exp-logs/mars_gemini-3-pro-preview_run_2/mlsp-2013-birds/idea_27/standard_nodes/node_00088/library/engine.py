import os
import time
import torch
import numpy as np
from library.config import Config
from library.utils import compute_auc, set_seed
from library.transforms import mixup_data
from library.losses import WeightedDistillationLoss


class Engine:
    """
    Encapsulates the training, validation, and prediction loops for the Bird Species Classification task.
    """

    def __init__(self, model, device, optimizer=None, scheduler=None, loss_fn=None):
        """
        Args:
            model (torch.nn.Module): The neural network model.
            device (str): Device to run training on ('cuda' or 'cpu').
            optimizer (torch.optim.Optimizer, optional): Optimizer.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
            loss_fn (torch.nn.Module, optional): Loss function (e.g., WeightedDistillationLoss).
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.best_score = -np.inf
        self.early_stopping_counter = 0

    def train_one_epoch(self, train_loader, epoch, mixup_alpha=0.4):
        """
        Trains the model for one epoch.
        Handles Mixup augmentation for both hard and soft targets (if present).
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            images = data["image"].to(self.device)
            targets = data["target"].to(self.device)
            soft_targets = data["soft_target"].to(self.device)

            # Combine targets for Mixup to ensure consistent permutation
            # targets: (B, C), soft_targets: (B, C) -> combined: (B, 2*C)
            # This allows us to mix both the hard labels and the soft distillation targets simultaneously
            combined_targets = torch.cat([targets, soft_targets], dim=1)

            # Apply Mixup
            images, targets_combined_a, targets_combined_b, lam = mixup_data(
                images, combined_targets, alpha=mixup_alpha, device=self.device
            )

            # Split back into hard and soft targets
            num_classes = targets.shape[1]
            targets_a = targets_combined_a[:, :num_classes]
            soft_targets_a = targets_combined_a[:, num_classes:]

            targets_b = targets_combined_b[:, :num_classes]
            soft_targets_b = targets_combined_b[:, num_classes:]

            self.optimizer.zero_grad()

            logits = self.model(images)

            # Compute Loss with Mixup
            # The loss function handles the weighting between hard and soft loss internally
            # We just need to mix the results based on the Mixup lambda
            loss_a = self.loss_fn(logits, targets_a, soft_targets_a)
            loss_b = self.loss_fn(logits, targets_b, soft_targets_b)
            loss = lam * loss_a + (1 - lam) * loss_b

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Computes average loss and AUC score.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for data in val_loader:
                images = data["image"].to(self.device)
                targets = data["target"].to(self.device)
                soft_targets = data["soft_target"].to(self.device)

                logits = self.model(images)

                # Validation loss (no mixup)
                loss = self.loss_fn(logits, targets, soft_targets)
                running_loss += loss.item()

                # Apply sigmoid for predictions
                preds = torch.sigmoid(logits)

                all_targets.append(targets.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        avg_loss = running_loss / len(val_loader)
        all_targets = np.concatenate(all_targets, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        auc_score = compute_auc(all_targets, all_preds)

        return avg_loss, auc_score

    def fit(
        self, train_loader, val_loader, epochs, patience=5, save_path="best_model.pth"
    ):
        """
        Runs the full training loop with early stopping.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
            save_path (str): Path to save the best model.

        Returns:
            float: The best validation AUC score achieved.
        """
        print(f"Starting training for {epochs} epochs with patience {patience}")

        # Ensure directory for save_path exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_one_epoch(
                train_loader, epoch, mixup_alpha=Config.MIXUP_ALPHA
            )
            val_loss, val_auc = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.10f} - "
                f"Val Loss: {val_loss:.10f} - "
                f"Val AUC: {val_auc:.10f} - "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpoint & Early Stopping
            # Cite debug_lesson_3: Sanitize Metrics and Initialize Bounds to Guarantee Checkpoint Creation
            current_score = 0.5 if np.isnan(val_auc) else val_auc

            if current_score > self.best_score:
                self.best_score = current_score
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model state before returning
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path))

        return self.best_score

    def predict(self, loader):
        """
        Generates predictions for a given loader.

        Args:
            loader: DataLoader.

        Returns:
            np.ndarray: Array of predicted probabilities (N, Num_Classes).
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for data in loader:
                images = data["image"].to(self.device)
                logits = self.model(images)
                preds = torch.sigmoid(logits)
                all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds, axis=0)
