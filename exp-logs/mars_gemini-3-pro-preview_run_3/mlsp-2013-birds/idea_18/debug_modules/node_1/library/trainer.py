import os
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import RobustAUC


class Trainer:
    """
    Manages the training lifecycle for a specific fold and architecture.
    Implements Gradient Accumulation, Mixup, and Top-K Checkpointing.
    """

    def __init__(self, model, optimizer, scheduler, device, fold_id, architecture_name):
        """
        Args:
            model (nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            device (str): Device to run on ('cuda' or 'cpu').
            fold_id (int): The current fold index.
            architecture_name (str): Name of the architecture (for checkpoint naming).
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.fold_id = fold_id
        self.arch_name = architecture_name

        # Multi-label classification loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Track best checkpoints: list of tuples (auc, filepath)
        self.best_checkpoints = []
        self.best_overall_auc = -float("inf")

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training with Mixup and Gradient Accumulation.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        # Zero grad at start of epoch
        self.optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            batch_size = images.size(0)

            # --- Mixup Regularization ---
            # We apply mixup to every batch as per strategy
            if Config.MIXUP_ALPHA > 0:
                lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            else:
                lam = 1.0

            if lam < 1.0:
                index = torch.randperm(batch_size).to(self.device)
                mixed_images = lam * images + (1 - lam) * images[index]
                # Mix targets for BCE loss
                mixed_labels = lam * labels + (1 - lam) * labels[index]

                outputs = self.model(mixed_images)
                loss = self.criterion(outputs, mixed_labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # --- Gradient Accumulation ---
            # Scale loss
            loss = loss / Config.ACCUMULATION_STEPS
            loss.backward()

            # Step optimizer only after accumulation steps
            if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Track unscaled loss for reporting (approximate)
            running_loss += loss.item() * Config.ACCUMULATION_STEPS * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set using RobustAUC.
        """
        self.model.eval()
        auc_metric = RobustAUC()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                # labels are already binary vectors
                labels = labels.to(self.device)

                outputs = self.model(images)
                # Apply sigmoid for probabilities
                probs = torch.sigmoid(outputs)

                auc_metric.update(probs, labels)

        return auc_metric.compute()

    def fit(self, train_loader, val_loader, num_epochs=Config.NUM_EPOCHS):
        """
        Runs the full training loop with early stopping and top-k checkpointing.

        Args:
            train_loader: DataLoader for training.
            val_loader: DataLoader for validation.
            num_epochs (int): Maximum number of epochs to train.

        Returns:
            list: Paths to the top-K checkpoints.
        """
        patience_counter = 0

        print(f"Starting training for {self.arch_name} - Fold {self.fold_id}")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_auc = self.validate(val_loader)

            # Step scheduler
            if self.scheduler:
                # Assuming ReduceLROnPlateau or similar, or just step() if Cosine
                # If it's ReduceLROnPlateau, it needs a metric.
                # If it's CosineAnnealing, it doesn't.
                # We'll assume standard StepLR/Cosine behavior unless it's Plateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_auc)
                else:
                    self.scheduler.step()

            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f} - Val AUC: {val_auc:.10f}"
            )

            # --- Checkpointing Logic (Top-K) ---
            # Save current model
            filename = f"{self.arch_name}_fold_{self.fold_id}_epoch_{epoch}.pth"
            save_path = os.path.join(Config.CHECKPOINT_DIR, filename)
            torch.save(self.model.state_dict(), save_path)

            # Add to list
            self.best_checkpoints.append((val_auc, save_path))

            # Sort descending by AUC
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

            # Keep only Top-K
            if len(self.best_checkpoints) > Config.TOP_K_CHECKPOINTS:
                # Remove the worst one from list and disk
                worst_auc, worst_path = self.best_checkpoints.pop()
                if os.path.exists(worst_path):
                    try:
                        os.remove(worst_path)
                    except OSError:
                        pass  # Handle potential race conditions or permission issues gracefully

            # --- Early Stopping Logic ---
            # Based on the absolute best AUC seen so far
            if val_auc > self.best_overall_auc:
                self.best_overall_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Return list of paths for the top-k checkpoints
        top_k_paths = [path for score, path in self.best_checkpoints]
        print(f"Training finished. Top-{len(top_k_paths)} checkpoints saved.")
        return top_k_paths
