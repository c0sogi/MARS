import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    laplace_log_likelihood_score,
    get_global_stats,
)
from library.data import get_dataloaders
from library.model import ARLRNet


class LaplaceNLLLoss(nn.Module):
    """
    Metric-Aligned Laplace Log Likelihood Loss.
    Optimizes in the standardized space to avoid gradient instability.
    Formula: L = (sqrt(2) * |y - mu|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super().__init__()
        self.sqrt_2 = torch.tensor(Config.SQRT_2)

    def forward(self, output, target):
        """
        Args:
            output: (B, 2) -> [mu_scaled, sigma_scaled]
            target: (B,) -> y_scaled
        """
        mu = output[:, 0]
        sigma = output[:, 1]  # Already constrained to be > epsilon in model

        # Ensure target shape matches mu
        if target.ndim == 1:
            target = target.view(-1)

        # Calculate NLL
        # Term 1: (sqrt(2) * |y - mu|) / sigma
        abs_diff = torch.abs(target - mu)
        term1 = (self.sqrt_2.to(sigma.device) * abs_diff) / sigma

        # Term 2: ln(sqrt(2) * sigma)
        term2 = torch.log(self.sqrt_2.to(sigma.device) * sigma)

        loss = term1 + term2
        return torch.mean(loss)


class Trainer:
    def __init__(self, model, train_loader, val_loader, global_stats):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.global_mean, self.global_std = global_stats
        self.device = torch.device(Config.DEVICE)
        self.criterion = LaplaceNLLLoss()

        self.model.to(self.device)
        self.optimizer = self._configure_optimizer()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        self.best_score = -float("inf")
        self.patience_counter = 0

    def _configure_optimizer(self):
        """
        Sets up AdamW with Differential Learning Rates.
        Lower LR for the pre-trained backbone, Higher LR for the rest.
        """
        backbone_params = []
        head_params = []

        # Identify backbone parameters by name or module
        # In ARLRNet, the backbone is inside stream_b.backbone
        for name, param in self.model.named_parameters():
            if "stream_b.backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.LR_BACKBONE},
                {"params": head_params, "lr": Config.LR_HEAD},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )
        return optimizer

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        for batch in self.train_loader:
            # Move data to device
            images = batch["image"].to(self.device)
            stream_a = batch["stream_a"].to(self.device)
            stream_b = batch["stream_b"].to(self.device)
            targets = batch["target"].to(self.device)

            # Forward pass
            outputs = self.model(images, stream_a, stream_b)

            # Loss calculation
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self):
        self.model.eval()
        val_loss_meter = AverageMeter()

        # Store predictions and targets for metric calculation
        all_mu_scaled = []
        all_sigma_scaled = []
        all_targets_raw = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                stream_a = batch["stream_a"].to(self.device)
                stream_b = batch["stream_b"].to(self.device)
                targets_scaled = batch["target"].to(self.device)
                targets_raw = batch["fvc_raw"].numpy()

                # Forward pass
                outputs = self.model(images, stream_a, stream_b)

                # Validation Loss (Standardized Space)
                loss = self.criterion(outputs, targets_scaled)
                val_loss_meter.update(loss.item(), images.size(0))

                # Collect outputs for Metric Calculation
                all_mu_scaled.append(outputs[:, 0].cpu().numpy())
                all_sigma_scaled.append(outputs[:, 1].cpu().numpy())
                all_targets_raw.append(targets_raw)

        # Concatenate
        mu_scaled = np.concatenate(all_mu_scaled)
        sigma_scaled = np.concatenate(all_sigma_scaled)
        y_true = np.concatenate(all_targets_raw)

        # Inverse Transform to Original Scale
        # mu_final = mu_scaled * sigma_global + mu_global
        # sigma_final = sigma_scaled * sigma_global
        mu_final = mu_scaled * self.global_std + self.global_mean
        sigma_final = sigma_scaled * self.global_std

        # Calculate Competition Metric
        # Note: laplace_log_likelihood_score handles the clipping (sigma>=70, error<=1000)
        metric_score = laplace_log_likelihood_score(y_true, mu_final, sigma_final)

        return val_loss_meter.avg, metric_score

    def fit(self):
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_score = self.validate()

            # Step Scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Time: {elapsed:.1f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Metric: {val_score:.16f}"
            )

            # Early Stopping & Checkpointing
            if val_score > self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"  >>> New Best Score! Model saved to {save_path}")
            else:
                self.patience_counter += 1
                print(f"  >>> Patience: {self.patience_counter}/{Config.PATIENCE}")

            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break


def run_training():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Global Stats for Normalization/Constraints
    global_mean, global_std = get_global_stats(Config.TRAIN_CSV)
    print(f"Global FVC Stats: Mean={global_mean:.4f}, Std={global_std:.4f}")

    # 4. Model
    print("Initializing ARLRNet...")
    model = ARLRNet(global_std_target=global_std)

    # 5. Training
    trainer = Trainer(model, train_loader, val_loader, (global_mean, global_std))
    trainer.fit()

    print(f"Training complete. Best Validation Metric: {trainer.best_score:.16f}")
