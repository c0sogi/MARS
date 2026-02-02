import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data import get_dataloaders
from library.model import HighCapacityCompositeModel


class MaskedL1Loss(nn.Module):
    """
    Computes L1 Loss restricted to the inspiratory phase (u_out == 0).
    Combines loss from the final head and the auxiliary head.
    """

    def __init__(self, aux_weight=0.3):
        super().__init__()
        self.aux_weight = aux_weight
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, final_pred, aux_pred, target, u_out):
        """
        Args:
            final_pred: (Batch, Time)
            aux_pred: (Batch, Time) or None
            target: (Batch, Time)
            u_out: (Batch, Time) - 0 for inspiration, 1 for expiration
        """
        # Create mask: 1 where u_out == 0 (inspiration), 0 otherwise
        mask = (u_out == 0).float()

        # Number of valid elements in the batch
        num_valid = mask.sum()

        # Avoid division by zero
        if num_valid == 0:
            return torch.tensor(0.0, device=final_pred.device, requires_grad=True)

        # Final Head Loss
        loss_final = (self.l1(final_pred, target) * mask).sum() / num_valid

        # Auxiliary Head Loss
        loss_aux = torch.tensor(0.0, device=final_pred.device)
        if aux_pred is not None:
            loss_aux = (self.l1(aux_pred, target) * mask).sum() / num_valid

        # Composite Loss
        total_loss = loss_final + self.aux_weight * loss_aux

        return total_loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, config=Config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device)

        self.model.to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler (OneCycleLR)
        # Steps per epoch is len(train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=config.pct_start,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
        )

        self.criterion = MaskedL1Loss(aux_weight=config.aux_loss_weight)
        self.best_val_mae = float("inf")

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            u_out = batch["u_out"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            final_pred, aux_pred = self.model(x)

            # Compute loss
            loss = self.criterion(final_pred, aux_pred, y, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        preds = []
        targets = []
        u_outs = []

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                u_out = batch["u_out"].to(self.device)

                # Forward pass (ignore auxiliary head for validation)
                final_pred, _ = self.model(x)

                # Store for metric computation
                preds.append(final_pred.cpu())
                targets.append(y.cpu())
                u_outs.append(u_out.cpu())

        # Concatenate all batches
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        u_outs = torch.cat(u_outs)

        # Compute MAE using the utility function
        val_mae = compute_metric(preds, targets, u_outs)
        return val_mae

    def fit(self):
        print(f"Starting training for {self.config.epochs} epochs on {self.device}...")

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_mae = self.validate()

            elapsed = time.time() - start_time

            # Logging (Full precision)
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val MAE: {val_mae:.16f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpoint
            if val_mae < self.best_val_mae:
                print(
                    f"Validation MAE improved from {self.best_val_mae:.16f} to {val_mae:.16f}. Saving model..."
                )
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), self.config.model_path)

        print(f"Training complete. Best Val MAE: {self.best_val_mae:.16f}")


def train_model():
    # 1. Reproducibility
    seed_everything(Config.seed)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        debug=Config.debug,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        load_cached_data=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = HighCapacityCompositeModel(config=Config)

    # 4. Trainer Initialization and Fitting
    trainer = Trainer(model, train_loader, val_loader, config=Config)
    trainer.fit()
