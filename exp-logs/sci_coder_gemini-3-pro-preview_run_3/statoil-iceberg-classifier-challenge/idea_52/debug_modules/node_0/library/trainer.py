import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import AverageMeter, save_checkpoint
from library.model import MS_IDPH_CNN


class Trainer:
    """
    Manages the training and validation process for the MS-IDPH-CNN model.
    """

    def __init__(self, model, device, train_loader, val_loader, fold_idx):
        """
        Args:
            model (nn.Module): The neural network model to train.
            device (torch.device): The device to run training on (CPU or GPU).
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            fold_idx (int): The current fold index for cross-validation logging/saving.
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold_idx = fold_idx

        # Optimizer: AdamW with constant learning rate and weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function: Binary Cross Entropy with Logits
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        loss_meter = AverageMeter()

        for batch in self.train_loader:
            # Unpack batch: images, angles, labels
            imgs, angs, labels = batch

            # Move to device
            imgs = imgs.to(self.device)
            angs = angs.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Model expects (x, angle)
            outputs = self.model(imgs, angs)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Update metrics
            loss_meter.update(loss.item(), imgs.size(0))

        return loss_meter.avg

    def validate(self):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        loss_meter = AverageMeter()

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, angs, labels = batch

                imgs = imgs.to(self.device)
                angs = angs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(imgs, angs)
                loss = self.criterion(outputs, labels)

                loss_meter.update(loss.item(), imgs.size(0))

        return loss_meter.avg

    def fit(self):
        """
        Runs the full training loop with Early Stopping and Checkpointing.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for Fold {self.fold_idx}...")

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Print metrics with full precision
            print(
                f"Fold {self.fold_idx} | Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Checkpointing and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    is_best=True,
                    checkpoint_dir=Config.CHECKPOINT_DIR,
                    filename=f"checkpoint_fold_{self.fold_idx}.pth",
                )

                # Also save explicitly as model_best_fold_X.pth for easy retrieval during inference
                best_model_path = os.path.join(
                    Config.CHECKPOINT_DIR, f"model_best_fold_{self.fold_idx}.pth"
                )
                torch.save(self.model.state_dict(), best_model_path)

            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {epoch+1} for Fold {self.fold_idx}"
                    )
                    break

        print(f"Fold {self.fold_idx} finished. Best Val Loss: {best_val_loss}")
        return best_val_loss
