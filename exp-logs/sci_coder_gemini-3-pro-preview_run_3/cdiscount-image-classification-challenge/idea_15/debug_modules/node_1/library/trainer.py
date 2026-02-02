import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything


class ModelTrainer:
    """
    Manages the training lifecycle of the Deep Feature Cascading model.
    Implements MixUp regularization, hierarchical loss calculation,
    validation scoring, and early stopping.
    """

    def __init__(self, model, train_loader, val_loader):
        """
        Args:
            model (nn.Module): The DeepFeatureCascade model.
            train_loader (DataLoader): Loader for training data.
            val_loader (DataLoader): Loader for validation data.
        """
        self.device = Config.DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler to reduce LR if validation metric plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        # Loss Function (Cross Entropy with Label Smoothing)
        # We use reduction='mean' by default
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Training State
        self.best_acc = 0.0
        self.start_epoch = 0

        seed_everything(Config.SEED)

    def mixup_data(self, x, l1, l2, l3, alpha=0.2):
        """
        Applies MixUp to the input features and returns mixed inputs and target pairs.

        Args:
            x (torch.Tensor): Input features.
            l1, l2, l3 (torch.Tensor): Target labels for each hierarchy level.
            alpha (float): MixUp hyperparameter.

        Returns:
            mixed_x: Mixed features.
            targets: Tuple containing (l1_a, l1_b, l2_a, l2_b, l3_a, l3_b).
            lam: Lambda mixing coefficient.
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]

        l1_a, l1_b = l1, l1[index]
        l2_a, l2_b = l2, l2[index]
        l3_a, l3_b = l3, l3[index]

        return mixed_x, (l1_a, l1_b, l2_a, l2_b, l3_a, l3_b), lam

    def mixup_criterion(self, preds, targets, lam):
        """
        Computes the MixUp loss summing across all hierarchy levels.

        Args:
            preds: Tuple of (pred_l1, pred_l2, pred_l3) logits.
            targets: Tuple of (l1_a, l1_b, l2_a, l2_b, l3_a, l3_b).
            lam: Lambda mixing coefficient.

        Returns:
            torch.Tensor: Combined loss.
        """
        pred_l1, pred_l2, pred_l3 = preds
        l1_a, l1_b, l2_a, l2_b, l3_a, l3_b = targets

        # Loss for Level 1
        loss_l1 = lam * self.criterion(pred_l1, l1_a) + (1 - lam) * self.criterion(
            pred_l1, l1_b
        )

        # Loss for Level 2
        loss_l2 = lam * self.criterion(pred_l2, l2_a) + (1 - lam) * self.criterion(
            pred_l2, l2_b
        )

        # Loss for Level 3
        loss_l3 = lam * self.criterion(pred_l3, l3_a) + (1 - lam) * self.criterion(
            pred_l3, l3_b
        )

        return loss_l1 + loss_l2 + loss_l3

    def train_epoch(self, epoch):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        start_time = time.time()

        for i, (features, l1, l2, l3) in enumerate(self.train_loader):
            features = features.to(self.device)
            l1 = l1.to(self.device)
            l2 = l2.to(self.device)
            l3 = l3.to(self.device)

            # Apply MixUp
            mixed_features, targets, lam = self.mixup_data(
                features, l1, l2, l3, alpha=Config.MIXUP_ALPHA
            )

            # Forward Pass
            self.optimizer.zero_grad()
            preds = self.model(mixed_features)

            # Compute Loss
            loss = self.mixup_criterion(preds, targets, lam)

            # Backward Pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        duration = time.time() - start_time
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f} | Time: {duration:.2f}s")
        return avg_loss

    def validate(self, epoch):
        """Runs validation and returns L3 accuracy."""
        self.model.eval()
        running_loss = 0.0
        correct_l3 = 0
        total = 0

        start_time = time.time()

        with torch.no_grad():
            for features, l1, l2, l3 in self.val_loader:
                features = features.to(self.device)
                l1 = l1.to(self.device)
                l2 = l2.to(self.device)
                l3 = l3.to(self.device)

                # Forward Pass (No MixUp)
                pred_l1, pred_l2, pred_l3 = self.model(features)

                # Calculate Standard Loss
                loss = (
                    self.criterion(pred_l1, l1)
                    + self.criterion(pred_l2, l2)
                    + self.criterion(pred_l3, l3)
                )
                running_loss += loss.item()

                # Calculate Accuracy for L3 (Target)
                _, predicted_l3 = torch.max(pred_l3.data, 1)
                total += l3.size(0)
                correct_l3 += (predicted_l3 == l3).sum().item()

        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct_l3 / total
        duration = time.time() - start_time

        print(
            f"Epoch {epoch} | Val Loss: {avg_loss:.6f} | Val Acc (L3): {accuracy} | Time: {duration:.2f}s"
        )

        return accuracy, avg_loss

    def train(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs.")
        print(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            # Train
            self.train_epoch(epoch)

            # Validate
            val_acc, val_loss = self.validate(epoch)

            # Scheduler Step
            self.scheduler.step(val_acc)

            # Checkpoint & Early Stopping
            if val_acc > self.best_acc:
                print(
                    f"Validation accuracy improved from {self.best_acc} to {val_acc}. Saving model..."
                )
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_acc}")
