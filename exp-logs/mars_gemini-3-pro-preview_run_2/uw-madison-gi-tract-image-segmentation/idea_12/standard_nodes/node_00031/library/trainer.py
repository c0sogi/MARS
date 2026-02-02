import os
import time
import torch
import numpy as np
from library.config import Config
from library.utils import set_seed, calculate_dice
from library.losses import BCEDiceLoss, TverskyLoss
from library.models import GhostUNet, EfficientNetUNet
from library.data_loader import get_dataloaders


class Trainer:
    """
    Trainer class to handle training and validation for both Coarse and Fine stages.
    """

    def __init__(self, stage):
        self.stage = stage
        self.device = Config.DEVICE

        # Ensure reproducibility
        set_seed(Config.SEED)

        # Initialize DataLoaders
        print(f"Initializing DataLoaders for stage: {stage}")
        self.train_loader, self.val_loader = get_dataloaders(stage)

        # Initialize Model, Loss, Optimizer, Scheduler based on stage
        if stage == "coarse":
            self.model = GhostUNet(
                in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES
            ).to(self.device)

            self.criterion = BCEDiceLoss(
                bce_weight=Config.COARSE_BCE_WEIGHT,
                dice_weight=Config.COARSE_DICE_WEIGHT,
            )

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=Config.COARSE_LR,
                weight_decay=Config.COARSE_WD,
            )
            self.epochs = Config.COARSE_EPOCHS
            self.save_path = Config.COARSE_MODEL_PATH

        elif stage == "fine":
            self.model = EfficientNetUNet(
                in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES
            ).to(self.device)

            self.criterion = TverskyLoss(
                alpha=Config.TVERSKY_ALPHA,
                beta=Config.TVERSKY_BETA,
                smooth=Config.TVERSKY_SMOOTH,
            )

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=Config.FINE_LR, weight_decay=Config.FINE_WD
            )
            self.epochs = Config.FINE_EPOCHS
            self.save_path = Config.FINE_MODEL_PATH
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        count = 0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Calculate Dice for monitoring
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                # Move to CPU for metric calculation using numpy utility
                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()

                batch_dice_sum = 0.0
                batch_size = preds_np.shape[0]

                for i in range(batch_size):
                    # Average dice over classes for this sample
                    sample_dice = 0.0
                    for c in range(Config.NUM_CLASSES):
                        sample_dice += calculate_dice(masks_np[i, c], preds_np[i, c])
                    batch_dice_sum += sample_dice / Config.NUM_CLASSES

                running_dice += batch_dice_sum
                count += batch_size

        epoch_loss = running_loss / len(self.train_loader)
        epoch_dice = running_dice / count

        return epoch_loss, epoch_dice

    def validate_one_epoch(self, epoch_idx):
        """
        Runs one epoch of validation.
        """
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        count = 0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()

                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                preds_np = preds.cpu().numpy()
                masks_np = masks.cpu().numpy()

                batch_dice_sum = 0.0
                batch_size = preds_np.shape[0]

                for i in range(batch_size):
                    sample_dice = 0.0
                    for c in range(Config.NUM_CLASSES):
                        sample_dice += calculate_dice(masks_np[i, c], preds_np[i, c])
                    batch_dice_sum += sample_dice / Config.NUM_CLASSES

                running_dice += batch_dice_sum
                count += batch_size

        epoch_loss = running_loss / len(self.val_loader)
        epoch_dice = running_dice / count

        return epoch_loss, epoch_dice

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {self.stage} model...")
        best_dice = 0.0
        patience = 5
        patience_counter = 0

        for epoch in range(self.epochs):
            start_time = time.time()

            train_loss, train_dice = self.train_one_epoch(epoch)
            val_loss, val_dice = self.validate_one_epoch(epoch)

            self.scheduler.step()

            duration = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{self.epochs} [{duration:.2f}s] "
                f"Train Loss: {train_loss}, Train Dice: {train_dice} | "
                f"Val Loss: {val_loss}, Val Dice: {val_dice}"
            )

            # Save Best Model
            if val_dice > best_dice:
                print(
                    f"Validation Dice improved from {best_dice} to {val_dice}. Saving model to {self.save_path}..."
                )
                best_dice = val_dice
                torch.save(self.model.state_dict(), self.save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training finished. Best Validation Dice: {best_dice}")


def run_training(stage):
    """
    Helper function to instantiate Trainer and run fit.
    """
    trainer = Trainer(stage)
    trainer.fit()
