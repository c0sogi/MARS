import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import NSLHN


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the negative of the modified Laplace Log Likelihood metric as a loss function.
    Minimizing this loss is equivalent to maximizing the competition metric.

    Formula: Loss = (sqrt(2) * Delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)
    where Delta = min(|True - Pred|, 1000) and sigma_clipped = max(sigma, 70).
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()

    def forward(self, y_pred, sigma, y_true):
        # Constants
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device))

        # Apply clipping constraints defined in the metric
        # Sigma clipped at 70ml
        sigma_clipped = torch.clamp(sigma, min=Config.CONFIDENCE_CLIP)

        # Absolute error clipped at 1000ml
        abs_error = torch.abs(y_true - y_pred)
        delta = torch.clamp(abs_error, max=Config.MAX_ERROR)

        # Calculate loss terms
        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        loss = term1 + term2
        return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        tabular = batch["tabular"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        week = batch["week"].to(device)
        base_week = batch["base_week"].to(device)
        y_true = batch["fvc"].to(device)

        optimizer.zero_grad()

        # Forward pass
        y_pred, sigma = model(axial, coronal, tabular, base_fvc, week, base_week)

        # Compute loss
        loss = criterion(y_pred, sigma, y_true)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Accumulate loss (weighted by batch size)
        running_loss += loss.item() * axial.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_one_epoch(model, loader, device):
    """
    Executes one validation epoch using the official metric.
    """
    model.eval()
    running_metric = 0.0

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            week = batch["week"].to(device)
            base_week = batch["base_week"].to(device)
            y_true = batch["fvc"].to(device)

            # Forward pass
            y_pred, sigma = model(axial, coronal, tabular, base_fvc, week, base_week)

            # Calculate metric using utility function
            # The utility returns the mean metric for the batch
            score = laplace_log_likelihood_metric(y_true, y_pred, sigma)

            # Accumulate weighted score
            running_metric += score * axial.size(0)

    epoch_metric = running_metric / len(loader.dataset)
    return epoch_metric


class Runner:
    """
    Manages the training, validation, and checkpointing process.
    """

    def __init__(self, debug=False):
        self.device = Config.DEVICE
        self.debug = debug

        # Initialize DataLoaders
        self.train_loader, self.val_loader, _ = get_dataloaders(debug=self.debug)

        # Initialize Model
        self.model = NSLHN().to(self.device)

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Loss Function
        self.criterion = LaplaceLogLikelihoodLoss()

        # Checkpointing
        self.best_score = -float("inf")
        self.best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    def train(self):
        print(f"Starting training on device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            # Training Step
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
            )

            # Validation Step
            val_score = validate_one_epoch(self.model, self.val_loader, self.device)

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging (Full precision for Val Score)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Score: {val_score} | "
                f"LR: {current_lr:.2e}"
            )

            # Early Stopping and Checkpointing
            if val_score > self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with score: {self.best_score}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Validation Score: {self.best_score}")


def main():
    seed_everything(Config.SEED)
    runner = Runner(debug=False)
    runner.train()
