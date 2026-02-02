import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import get_dataloaders
from library.model import VentilatorModel


class MaskedL1Loss(nn.Module):
    """
    Computes L1 Loss masked by the inspiratory phase (u_out == 0).
    """

    def forward(self, pred, target, u_out):
        # u_out: 0 = inspiratory, 1 = expiratory
        # Mask: 1 for inspiratory, 0 for expiratory
        mask = 1 - u_out

        # Calculate element-wise absolute error
        loss = torch.abs(pred - target) * mask

        # Average over the number of valid elements (inspiratory steps)
        # Add epsilon to avoid division by zero
        return loss.sum() / (mask.sum() + 1e-8)


class Trainer:
    """
    Manages the training, validation, and checkpointing process.
    """

    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE

        # Move model to device
        self.model.to(self.device)

        # Loss function
        self.criterion = MaskedL1Loss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR_MAX, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler (OneCycleLR)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LR_MAX,
            steps_per_epoch=len(train_loader),
            epochs=Config.EPOCHS,
            pct_start=Config.PCT_START,
            div_factor=Config.DIV_FACTOR,
            final_div_factor=Config.FINAL_DIV_FACTOR,
        )

        self.best_mae = float("inf")

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            # Move batch to device
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            u_out = batch["u_out"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # Model returns (final_out, aux_out)
            final_out, aux_out = self.model(x)

            # Squeeze last dimension: (Batch, Seq, 1) -> (Batch, Seq)
            final_out = final_out.squeeze(-1)

            # Calculate Main Loss
            loss = self.criterion(final_out, y, u_out)

            # Calculate Auxiliary Loss if available
            if aux_out is not None:
                aux_out = aux_out.squeeze(-1)
                loss_aux = self.criterion(aux_out, y, u_out)
                loss += Config.AUX_WEIGHT * loss_aux

            # Backward pass
            loss.backward()

            # Strict Gradient Clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                u_out = batch["u_out"].to(self.device)

                # Forward pass
                final_out, _ = self.model(x)
                final_out = final_out.squeeze(-1)

                # Compute MAE metric
                mae = compute_metric(final_out, y, u_out)
                total_mae += mae
                num_batches += 1

        return total_mae / num_batches

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch()
            val_mae = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val MAE: {val_mae}"
            )

            # Checkpointing
            if val_mae < self.best_mae:
                self.best_mae = val_mae
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"New best model saved with MAE: {self.best_mae}")


def run_training():
    """
    Entry point function to setup and run the training process.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Load DataLoaders (using caching mechanism)
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Initialize Model
    # Input dimension is automatically inferred by the model class based on Config
    model = VentilatorModel()

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader)

    # Start Training
    trainer.fit()
