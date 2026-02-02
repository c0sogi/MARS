import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    metric_laplace_log_likelihood,
    save_submission_file,
)
from library.data import get_dataloaders
from library.model import DSPRNet


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Metric-Aligned Laplace Log Likelihood Loss.
    Formula: L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, mu, sigma, target):
        # Ensure device consistency
        if self.sqrt_2.device != mu.device:
            self.sqrt_2 = self.sqrt_2.to(mu.device)

        # Reshape target to match output (B, 1) if necessary
        if target.ndim == 1:
            target = target.view(-1, 1)
        if mu.ndim == 1:
            mu = mu.view(-1, 1)
        if sigma.ndim == 1:
            sigma = sigma.view(-1, 1)

        abs_diff = torch.abs(target - mu)
        loss = (self.sqrt_2 * abs_diff) / sigma + torch.log(self.sqrt_2 * sigma)
        return loss.mean()


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, scalers):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scalers = scalers
        self.device = Config.DEVICE

        self.criterion = LaplaceLogLikelihoodLoss()

        # Differential Learning Rates
        backbone_params = list(self.model.image_encoder.backbone.parameters())
        backbone_ids = list(map(id, backbone_params))
        head_params = filter(
            lambda p: id(p) not in backbone_ids, self.model.parameters()
        )

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.LR_BACKBONE},
                {"params": head_params, "lr": Config.LR_HEAD},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        self.best_score = -float("inf")
        self.early_stopping_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        for batch_idx, ((images, clinical), targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            clinical = clinical.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            mu, sigma = self.model(images, clinical)
            loss = self.criterion(mu, sigma, targets)

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self):
        self.model.eval()

        all_mu = []
        all_sigma = []
        all_targets = []

        with torch.no_grad():
            for (images, clinical), targets in self.val_loader:
                images = images.to(self.device)
                clinical = clinical.to(self.device)

                mu, sigma = self.model(images, clinical)

                all_mu.append(mu.cpu().numpy())
                all_sigma.append(sigma.cpu().numpy())
                all_targets.append(targets.numpy())

        all_mu = np.concatenate(all_mu)
        all_sigma = np.concatenate(all_sigma)
        all_targets = np.concatenate(all_targets)

        # Inverse Transform for Metric Calculation
        if self.scalers and "target" in self.scalers:
            scaler = self.scalers["target"]
            # Inverse transform mean
            # scaler expects (N, 1)
            mu_real = scaler.inverse_transform(all_mu.reshape(-1, 1)).flatten()
            target_real = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
            # Inverse transform std dev (scale only)
            sigma_real = all_sigma.flatten() * scaler.scale_[0]
        else:
            mu_real = all_mu.flatten()
            target_real = all_targets.flatten()
            sigma_real = all_sigma.flatten()

        # Calculate Metric
        score = metric_laplace_log_likelihood(target_real, mu_real, sigma_real)
        return score

    def fit(self):
        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_score = self.validate()
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.6f}"
            )

            # Early Stopping & Checkpointing
            if val_score > self.best_score:
                self.best_score = val_score
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                # print(f"  New best model saved! Score: {self.best_score:.6f}")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Val Score: {self.best_score:.6f}")

    def predict(self):
        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        all_mu = []
        all_sigma = []

        with torch.no_grad():
            for (images, clinical), _ in self.test_loader:
                images = images.to(self.device)
                clinical = clinical.to(self.device)

                mu, sigma = self.model(images, clinical)

                all_mu.append(mu.cpu().numpy())
                all_sigma.append(sigma.cpu().numpy())

        all_mu = np.concatenate(all_mu)
        all_sigma = np.concatenate(all_sigma)

        # Inverse Transform
        if self.scalers and "target" in self.scalers:
            scaler = self.scalers["target"]
            mu_real = scaler.inverse_transform(all_mu.reshape(-1, 1)).flatten()
            sigma_real = all_sigma.flatten() * scaler.scale_[0]
        else:
            mu_real = all_mu.flatten()
            sigma_real = all_sigma.flatten()

        # Apply Post-Processing Clip for Submission
        # "Confidence values are clipped at 70 ml"
        sigma_clipped = np.maximum(sigma_real, Config.METRIC_CLIP_SIGMA)

        # Construct Submission DataFrame
        # We need Patient_Week identifiers. These are stored in the dataset df.
        test_df = self.test_loader.dataset.df

        submission_df = pd.DataFrame(
            {
                "Patient_Week": test_df["Patient_Week"],
                "FVC": mu_real,
                "Confidence": sigma_clipped,
            }
        )

        # Save
        save_submission_file(submission_df)
        print(f"Submission generated at {Config.SUBMISSION_PATH}")


def train():
    seed_everything(Config.SEED)

    # Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader, scalers = get_dataloaders(
        load_cached_data=True
    )

    # Initialize Model
    model = DSPRNet()

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, scalers)

    # Train
    trainer.fit()

    # Predict
    trainer.predict()


if __name__ == "__main__":
    pass  # Managed by external execution
