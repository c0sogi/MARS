import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, seed_everything
from library.dataset import LungFVCDataset
from library.model import DualPathTransformer
from library.utils import laplace_log_likelihood_metric, unscale_data


class Trainer:
    """
    Manages the training and validation lifecycle of the Fused MLP Network.
    """

    def __init__(self, debug=False):
        """
        Args:
            debug (bool): If True, uses a subset of data for faster debugging.
        """
        self.device = torch.device(Config.DEVICE)
        self.debug = debug

        # Initialize Model
        self.model = DualPathTransformer().to(self.device)

        # Initialize Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Best score tracker (Metric is negative, higher is better)
        self.best_score = -float("inf")

    def get_dataloaders(self):
        """
        Loads metadata and creates DataLoaders for training and validation.
        """
        # Load Metadata
        train_df = pd.read_csv(Config.TRAIN_META_PATH)
        val_df = pd.read_csv(Config.VAL_META_PATH)

        # Initialize Datasets
        train_dataset = LungFVCDataset(train_df, mode="train", debug=self.debug)
        val_dataset = LungFVCDataset(val_df, mode="val", debug=self.debug)

        # Initialize DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader

    def negative_log_likelihood_loss(self, fvc_pred, sigma_pred, target):
        """
        Differentiable NLL Loss for training.
        Loss = |y - y_hat| / sigma + log(sigma)

        Note: Inputs are expected to be standardized (Z-scores) to match model output range.
        """
        # Add epsilon to sigma to prevent log(0) or division by zero,
        # though model output softplus+1e-3 handles this.
        sigma_pred = torch.clamp(sigma_pred, min=1e-6)

        # Calculate NLL
        loss = torch.abs(target - fvc_pred) / sigma_pred + torch.log(sigma_pred)

        return torch.mean(loss)

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Move data to device
            images = batch["images"].to(self.device)
            meta_age = batch["meta_age"].to(self.device)
            meta_sex = batch["meta_sex"].to(self.device)
            meta_smoke = batch["meta_smoke"].to(self.device)
            linear_features = batch["linear_features"].to(self.device)
            target = batch["target"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            fvc_pred, sigma_pred = self.model(
                images, meta_age, meta_sex, meta_smoke, linear_features
            )

            # Calculate loss
            loss = self.negative_log_likelihood_loss(fvc_pred, sigma_pred, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader):
        """
        Runs validation and calculates the competition metric.
        """
        self.model.eval()
        all_true_fvc = []
        all_pred_fvc = []
        all_pred_sigma = []

        with torch.no_grad():
            for batch in val_loader:
                # Move inputs to device
                images = batch["images"].to(self.device)
                meta_age = batch["meta_age"].to(self.device)
                meta_sex = batch["meta_sex"].to(self.device)
                meta_smoke = batch["meta_smoke"].to(self.device)
                linear_features = batch["linear_features"].to(self.device)

                # Forward pass
                fvc_pred_std, sigma_pred_std = self.model(
                    images, meta_age, meta_sex, meta_smoke, linear_features
                )

                # Unscale predictions to original units (ml)
                fvc_pred, sigma_pred = unscale_data(fvc_pred_std, sigma_pred_std)

                # Get true FVC (dataset returns standardized target, we need raw for metric)
                # We can reconstruct raw from standardized or fetch from dataframe,
                # but dataset batch doesn't strictly carry raw FVC.
                # However, unscaling the target tensor works.
                target_std = batch["target"].to(self.device)
                fvc_true, _ = unscale_data(target_std, torch.zeros_like(target_std))

                # Store results
                all_true_fvc.append(fvc_true.cpu().numpy())
                all_pred_fvc.append(fvc_pred.cpu().numpy())
                all_pred_sigma.append(sigma_pred.cpu().numpy())

        # Concatenate all batches
        y_true = np.concatenate(all_true_fvc)
        y_pred = np.concatenate(all_pred_fvc)
        sigma = np.concatenate(all_pred_sigma)

        # Calculate official metric
        score = laplace_log_likelihood_metric(y_true, y_pred, sigma)
        return score

    def fit(self, epochs=Config.EPOCHS):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")
        train_loader, val_loader = self.get_dataloaders()

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_score = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()

            # Checkpoint
            if val_score > self.best_score:
                self.best_score = val_score
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(
                    f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Score: {val_score} | Saved Best Model"
                )
            else:
                print(
                    f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Score: {val_score}"
                )

        print(f"Training complete. Best Validation Score: {self.best_score}")
