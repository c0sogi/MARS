import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from timm.loss import SoftTargetCrossEntropy

from library.config import Config
from library.utils import seed_everything
from library.model import IWildCamModel
from library.data_loader import get_dataloaders


class Trainer:
    """
    Trainer class to manage the training and validation of the IWildCam model.
    Implements the 'A3' recipe: AdamW, AMP, and regularized loss.
    """

    def __init__(self, debug=False):
        """
        Initialize the Trainer.

        Args:
            debug (bool): If True, runs with a smaller subset of data for debugging.
        """
        self.debug = debug
        self.device = Config.DEVICE

        # Ensure reproducibility
        seed_everything(Config.SEED)

        # 1. Data Loaders
        # mixup_fn is returned by get_dataloaders based on Config parameters
        self.train_loader, self.val_loader, self.test_loader, self.mixup_fn = (
            get_dataloaders(debug=self.debug)
        )

        # 2. Model
        print(f"Initializing model: {Config.MODEL_NAME}")
        self.model = IWildCamModel(
            model_name=Config.MODEL_NAME,
            num_classes=Config.NUM_CLASSES,
            pretrained=True,
        )
        self.model.to(self.device)

        # 3. Optimization
        # AdamW is standard for EfficientNetV2 and Transformers
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Cosine Annealing Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler("cuda")

        # 4. Loss Functions
        # SoftTargetCrossEntropy is required for Mixup (targets are probabilities)
        # Label smoothing is handled within Mixup or the loss depending on implementation,
        # but timm's Mixup usually outputs soft labels, so SoftTargetCrossEntropy is the match.
        self.train_criterion = SoftTargetCrossEntropy()

        # Standard CrossEntropy for validation (targets are indices)
        self.val_criterion = nn.CrossEntropyLoss()

        # Early Stopping parameters
        self.best_val_acc = 0.0
        self.patience = 5
        self.counter = 0

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Apply Mixup/CutMix
            # mixup_fn returns (mixed_images, mixed_targets_soft)
            if self.mixup_fn is not None:
                images, targets = self.mixup_fn(images, targets)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast("cuda"):
                outputs = self.model(images)
                loss = self.train_criterion(outputs, targets)

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()

            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        duration = time.time() - start_time

        print(
            f"Epoch {epoch_idx+1}/{Config.EPOCHS} | Train Loss: {epoch_loss} | Time: {duration}s"
        )
        return epoch_loss

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Standard forward pass (no mixup, no autocast needed for val but safe to use)
                with autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.val_criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)

                # Calculate Accuracy
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += images.size(0)

        val_loss = running_loss / total_samples
        val_acc = correct_predictions / total_samples

        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_acc}")

        return val_loss, val_acc

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print("Starting training...")

        for epoch in range(Config.EPOCHS):
            # Train
            self.train_one_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Step Scheduler
            self.scheduler.step()

            # Checkpointing & Early Stopping
            if val_acc > self.best_val_acc:
                print(
                    f"Validation accuracy improved from {self.best_val_acc} to {val_acc}. Saving model..."
                )
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                self.counter = 0
            else:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Accuracy: {self.best_val_acc}")
