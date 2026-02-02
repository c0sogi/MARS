import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, score
from library.data import get_dataloaders
from library.model import MPVERNet


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Differentiable implementation of the modified Laplace Log Likelihood loss.
    Loss = -Metric
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        # Constants from Config/Task
        MAX_ERROR = float(Config.MAX_ERROR)
        MIN_SIGMA = float(Config.MIN_CONFIDENCE)
        SQRT_2 = torch.sqrt(torch.tensor(2.0, device=pred_fvc.device))

        # 1. Calculate absolute error (delta)
        delta = torch.abs(true_fvc - pred_fvc)

        # 2. Clip error at 1000 ml to filter outliers (as per task description)
        # This acts as a robust regression component
        delta = torch.clamp(delta, max=MAX_ERROR)

        # 3. Clip confidence at 70 ml
        sigma_clipped = torch.clamp(pred_sigma, min=MIN_SIGMA)

        # 4. Calculate Loss (Negative Log Likelihood)
        # Metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        # Loss = -Metric = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)

        term1 = (SQRT_2 * delta) / sigma_clipped
        term2 = torch.log(SQRT_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = LaplaceLogLikelihoodLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        self.best_score = -float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            # Move data to device
            img_ax = batch["image_axial"].to(self.device)
            img_cor = batch["image_coronal"].to(self.device)
            tab_norm = batch["tabular_norm"].to(self.device)
            tab_raw = batch["tabular_raw"].to(self.device)
            time_delta = batch["time_delta"].to(self.device)
            target = batch["target"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            pred_fvc, pred_sigma = self.model(
                img_ax, img_cor, tab_norm, tab_raw, time_delta
            )

            # Loss calculation
            loss = self.criterion(pred_fvc, pred_sigma, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * img_ax.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        self.model.eval()
        all_true = []
        all_pred = []
        all_sigma = []

        with torch.no_grad():
            for batch in self.val_loader:
                img_ax = batch["image_axial"].to(self.device)
                img_cor = batch["image_coronal"].to(self.device)
                tab_norm = batch["tabular_norm"].to(self.device)
                tab_raw = batch["tabular_raw"].to(self.device)
                time_delta = batch["time_delta"].to(self.device)
                target = batch["target"].to(self.device)

                pred_fvc, pred_sigma = self.model(
                    img_ax, img_cor, tab_norm, tab_raw, time_delta
                )

                all_true.append(target.cpu().numpy())
                all_pred.append(pred_fvc.cpu().numpy())
                all_sigma.append(pred_sigma.cpu().numpy())

        all_true = np.concatenate(all_true, axis=0)
        all_pred = np.concatenate(all_pred, axis=0)
        all_sigma = np.concatenate(all_sigma, axis=0)

        # Calculate official metric
        val_score = score(all_true, all_pred, all_sigma)
        return val_score

    def fit(self):
        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_score = self.validate()

            # Step Scheduler
            self.scheduler.step()

            # Print metrics (Full precision as requested)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Score: {val_score}"
            )

            # Early Stopping and Checkpointing
            if val_score > self.best_score:
                self.best_score = val_score
                self.patience_counter = 0

                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered after {Config.PATIENCE} epochs without improvement."
                    )
                    break

        print(f"Training complete. Best Validation Score: {self.best_score}")


def train_model():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders()

    # 3. Model
    print("Initializing MPVERNet...")
    model = MPVERNet()
    model.to(device)

    # 4. Trainer
    trainer = Trainer(model, train_loader, val_loader, device)

    # 5. Execute
    trainer.fit()


if __name__ == "__main__":
    train_model()
