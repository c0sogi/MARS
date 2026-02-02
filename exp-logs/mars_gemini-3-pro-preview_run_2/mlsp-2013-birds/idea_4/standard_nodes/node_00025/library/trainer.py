import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.utils import calculate_metric


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer:
    def __init__(self, config, model, train_loader, val_loader, fold_idx=0):
        """
        Initializes the Trainer.

        Args:
            config: Configuration object.
            model: The PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            fold_idx: Index of the current fold (for saving models).
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold_idx = fold_idx
        self.device = torch.device(config.device)

        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6
        )

        # Loss Function with Class Imbalance Handling
        pos_weight = self._calculate_pos_weight()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Early Stopping parameters
        self.best_val_auc = 0.0
        self.patience = 15
        self.counter = 0

    def _calculate_pos_weight(self):
        """
        Calculates positive weights for BCEWithLogitsLoss based on training data.
        Weight = (Number of Negatives) / (Number of Positives)
        """
        # Access labels from the dataset
        # Assuming dataset.labels is a numpy array of shape (N, num_classes)
        labels = self.train_loader.dataset.labels

        # Calculate counts
        pos_counts = np.sum(labels, axis=0)
        total_counts = len(labels)
        neg_counts = total_counts - pos_counts

        # Avoid division by zero for classes that might not appear in a specific fold
        # If a class has 0 positives, the weight doesn't strictly matter for the positive term,
        # but we set it to a safe value.
        safe_pos_counts = np.maximum(pos_counts, 1)

        weights = neg_counts / safe_pos_counts

        # Convert to tensor
        return torch.FloatTensor(weights).to(self.device)

    def train_one_epoch(self, epoch):
        """
        Trains the model for one epoch using Mixup.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # Apply Mixup
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, self.config.mixup_alpha, self.device
            )

            # Forward pass
            outputs = self.model(images)
            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate_one_epoch(self, epoch):
        """
        Validates the model and calculates ROC AUC.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.size(0)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply sigmoid for metric calculation
                probs = torch.sigmoid(outputs)

                all_preds.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        if dataset_size == 0:
            return 0.0, 0.5

        epoch_loss = running_loss / dataset_size

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Metric
        val_auc = calculate_metric(all_targets, all_preds)

        return epoch_loss, val_auc

    def fit(self):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training for Fold {self.fold_idx}...")

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_auc = self.validate_one_epoch(epoch)

            # Scheduler Step
            self.scheduler.step()

            end_time = time.time()
            duration = end_time - start_time

            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc}"
            )

            # Early Stopping and Checkpointing
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.counter = 0
                self._save_model()
                print(f"New best model found! Saved to {self._get_model_path()}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Best Val AUC for Fold {self.fold_idx}: {self.best_val_auc}")
        return self.best_val_auc

    def _get_model_path(self):
        filename = f"model_fold_{self.fold_idx}.pth"
        return os.path.join(self.config.model_output_dir, filename)

    def _save_model(self):
        torch.save(self.model.state_dict(), self._get_model_path())
