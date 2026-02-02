import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import save_checkpoint
from library.model import (
    EfficientNetAudio,
    train_one_epoch,
    validate,
    predict_and_submit,
)


class Trainer:
    """
    Trainer class to manage the training and validation loops for Speech Commands classification.
    """

    def __init__(self, device=Config.DEVICE):
        """
        Initialize the Trainer with model, optimizer, scheduler, and criterion.

        Args:
            device (torch.device): The device to run training on.
        """
        self.device = device

        # Initialize Model
        self.model = EfficientNetAudio(num_classes=Config.NUM_CLASSES)
        self.model.to(self.device)

        # Initialize Criterion
        self.criterion = nn.CrossEntropyLoss()

        # Initialize Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler (CosineAnnealingLR)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS, patience=10):
        """
        Runs the training loop with Early Stopping and Scheduler.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Maximum number of epochs to train.
            patience (int): Early stopping patience.

        Returns:
            float: Best validation accuracy achieved.
        """
        best_acc = 0.0
        patience_counter = 0
        best_epoch = 0

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            # Train for one epoch
            train_loss = train_one_epoch(
                self.model,
                train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                Config.MIXUP_ALPHA,
            )

            # Validate
            val_loss, val_acc = validate(
                self.model, val_loader, self.criterion, self.device
            )

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}: LR={current_lr}, Train Loss={train_loss}, Val Loss={val_loss}, Val Acc={val_acc}"
            )

            # Checkpoint & Early Stopping Logic
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                best_epoch = epoch
                patience_counter = 0

                # Save Checkpoint
                state = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }
                save_checkpoint(state, is_best=True, checkpoint_dir=Config.WORKING_DIR)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best Acc: {best_acc} at epoch {best_epoch}"
                )
                break

        print(f"Training complete. Best Validation Accuracy: {best_acc}")
        return best_acc

    def predict(self, test_loader, output_path=Config.SUBMISSION_PATH):
        """
        Generates predictions for the test set and saves to CSV.
        Uses the best model saved during training (handled by predict_and_submit).

        Args:
            test_loader (DataLoader): DataLoader for test data.
            output_path (str): Path to save the submission CSV.
        """
        predict_and_submit(
            self.model, test_loader, device=self.device, output_path=output_path
        )
