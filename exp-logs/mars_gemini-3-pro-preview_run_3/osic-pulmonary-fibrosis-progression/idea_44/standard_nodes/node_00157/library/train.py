import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from library.config import Config
from library.utils import seed_everything, MetricMonitor, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import GPCRNet


class LaplaceLoss(nn.Module):
    """
    Multi-Task Metric-Aligned Laplace Log Likelihood Loss.
    Computes loss for both main and auxiliary predictions.
    """

    def __init__(self):
        super(LaplaceLoss, self).__init__()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tuple ((mu, sigma), (aux_mu, aux_sigma))
            targets: Tensor of shape (Batch,) containing scaled ground truth FVC.
        """
        (mu, sigma), (aux_mu, aux_sigma) = outputs

        # Calculate Main Loss
        # L = (sqrt(2) * |y - mu|) / sigma + ln(sqrt(2) * sigma)
        delta = torch.abs(targets - mu)
        sqrt_2 = math.sqrt(2)

        # Add small epsilon to sigma inside log to prevent NaN if sigma is extremely small
        # (though softplus + 1e-6 in model should prevent this)
        main_loss = (sqrt_2 * delta) / sigma + torch.log(sqrt_2 * sigma)
        main_loss = torch.mean(main_loss)

        # Calculate Auxiliary Loss
        aux_delta = torch.abs(targets - aux_mu)
        aux_loss = (sqrt_2 * aux_delta) / aux_sigma + torch.log(sqrt_2 * aux_sigma)
        aux_loss = torch.mean(aux_loss)

        # Total Loss
        total_loss = main_loss + 0.5 * aux_loss

        return total_loss


class Runner:
    """
    Manages the training and validation lifecycle of the GPCR-Net.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        seed_everything(Config.SEED)

        # Initialize Model
        self.model = GPCRNet().to(self.device)

        # Initialize Loss
        self.criterion = LaplaceLoss()

        # Initialize Optimizer with Differential Learning Rates
        # Group 1: Backbone (Pretrained) -> LR_BACKBONE
        # Group 2: New Layers (Projection, Streams, Heads) -> LR_HEAD

        backbone_ids = list(map(id, self.model.image_encoder.backbone.parameters()))
        base_params = filter(lambda p: id(p) in backbone_ids, self.model.parameters())
        head_params = filter(
            lambda p: id(p) not in backbone_ids, self.model.parameters()
        )

        self.optimizer = optim.AdamW(
            [
                {"params": base_params, "lr": Config.LR_BACKBONE},
                {"params": head_params, "lr": Config.LR_HEAD},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX
        )

        # Metrics and Checkpointing
        self.best_score = -float("inf")
        self.early_stopping_patience = 10
        self.patience_counter = 0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        loss_monitor = MetricMonitor()

        for batch_idx, (images, clinical, targets) in enumerate(train_loader):
            images = images.to(self.device)
            clinical = clinical.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, clinical)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            loss_monitor.update(loss.item(), images.size(0))

        return loss_monitor.avg

    def validate(self, val_loader):
        self.model.eval()

        # Store predictions and targets for metric calculation
        all_mu = []
        all_sigma = []
        all_targets = []

        with torch.no_grad():
            for images, clinical, targets in val_loader:
                images = images.to(self.device)
                clinical = clinical.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                (mu, sigma), _ = self.model(images, clinical)

                # Collect results (keep on CPU for numpy processing)
                all_mu.append(mu.cpu().numpy())
                all_sigma.append(sigma.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Concatenate batches
        mu_scaled = np.concatenate(all_mu)
        sigma_scaled = np.concatenate(all_sigma)
        targets_scaled = np.concatenate(all_targets)

        # Inverse Transform to Original Scale (ml)
        # mu_orig = mu_scaled * std + mean
        # sigma_orig = sigma_scaled * std
        # target_orig = target_scaled * std + mean

        mu_orig = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
        sigma_orig = sigma_scaled * Config.TARGET_STD
        targets_orig = targets_scaled * Config.TARGET_STD + Config.TARGET_MEAN

        # Calculate Official Metric
        score = laplace_log_likelihood_metric(targets_orig, mu_orig, sigma_orig)

        return score

    def train(self, debug=False):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")

        # Get DataLoaders
        # If debug is True, we could potentially use a subset, but here we use full data
        # as controlled by the config/metadata.
        train_loader, val_loader = get_dataloaders()

        for epoch in range(1, Config.EPOCHS + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_score = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.8f} | Val Score: {val_score:.8f}"
            )

            # Checkpoint & Early Stopping
            if val_score > self.best_score:
                print(
                    f"Score Improved ({self.best_score:.8f} -> {val_score:.8f}). Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best Validation Score: {self.best_score:.8f}")


def run_training():
    runner = Runner()
    runner.train()
