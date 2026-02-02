import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from library.config import Config
from library.utils import LaplaceLogLikelihoodLoss, score
from library.data import get_dataloaders
from library.model import AASLNet


class Trainer:
    """
    Manages the training, validation, and early stopping logic for AASL-Net.
    """

    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(Config.DEVICE)

        # Move model to appropriate device
        self.model.to(self.device)

        # Loss Function
        self.criterion = LaplaceLogLikelihoodLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Early Stopping parameters
        self.patience = Config.PATIENCE
        self.best_score = -float("inf")  # Metric is negative, higher is better
        self.counter = 0
        self.best_model_path = Config.MODEL_SAVE_PATH

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (img_ax, img_cor, tabular, meta, target) in enumerate(
            self.train_loader
        ):
            # Move data to device
            img_ax = img_ax.to(self.device)
            img_cor = img_cor.to(self.device)
            tabular = tabular.to(self.device)
            meta = meta.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred_fvc, pred_sigma = self.model(img_ax, img_cor, tabular, meta)

            # Loss calculation
            loss = self.criterion(pred_fvc, pred_sigma, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set and computes the competition metric.
        """
        self.model.eval()

        all_pred_fvc = []
        all_pred_sigma = []
        all_true_fvc = []

        with torch.no_grad():
            for img_ax, img_cor, tabular, meta, target in self.val_loader:
                img_ax = img_ax.to(self.device)
                img_cor = img_cor.to(self.device)
                tabular = tabular.to(self.device)
                meta = meta.to(self.device)

                pred_fvc, pred_sigma = self.model(img_ax, img_cor, tabular, meta)

                # Collect results for metric calculation
                all_pred_fvc.append(pred_fvc.cpu().numpy())
                all_pred_sigma.append(pred_sigma.cpu().numpy())
                all_true_fvc.append(target.cpu().numpy())

        # Concatenate all batches
        all_pred_fvc = np.concatenate(all_pred_fvc)
        all_pred_sigma = np.concatenate(all_pred_sigma)
        all_true_fvc = np.concatenate(all_true_fvc)

        # Calculate metric using the provided utility
        val_score = score(all_pred_fvc, all_pred_sigma, all_true_fvc)
        return val_score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_score = self.validate()

            # Step the scheduler
            self.scheduler.step()

            # Print metrics (Full precision for validation score as requested)
            print(
                f"Epoch {epoch}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Score: {val_score}"
            )

            # Early Stopping Logic
            if val_score > self.best_score:
                self.best_score = val_score
                self.counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with score: {val_score}")
            else:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Score: {self.best_score}")
        return self.best_score


def run_training(debug=False):
    """
    Initializes the model, data loaders, and trainer, then starts the training process.

    Args:
        debug (bool): If True, uses a small subset of data for quick debugging.

    Returns:
        float: The best validation score achieved.
    """
    # Load data
    train_loader, val_loader = get_dataloaders(debug=debug)

    # Initialize model
    model = AASLNet()

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader)

    # Run training
    best_score = trainer.fit()

    return best_score
