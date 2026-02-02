import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.model import DSPRNet
from library.data import get_dataloaders
from library.utils import seed_everything, InverseScaler, LaplaceMetric


class MetricAlignedLoss(nn.Module):
    """
    Implements the Metric-Aligned Laplace Log Likelihood Loss.
    Loss = (sqrt(2) * |target - pred|) / sigma + log(sqrt(2) * sigma)
    """

    def __init__(self):
        super().__init__()
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, pred_fvc, pred_sigma, target_fvc):
        """
        Args:
            pred_fvc: Scaled predicted FVC.
            pred_sigma: Scaled predicted confidence (must be positive).
            target_fvc: Scaled true FVC.
        """
        # Calculate absolute error
        delta = torch.abs(target_fvc - pred_fvc)

        # Calculate loss terms
        # Term 1: (sqrt(2) * delta) / sigma
        term1 = (self.sqrt_2.to(pred_fvc.device) * delta) / pred_sigma

        # Term 2: log(sqrt(2) * sigma)
        term2 = torch.log(self.sqrt_2.to(pred_fvc.device) * pred_sigma)

        loss = torch.mean(term1 + term2)
        return loss


class Trainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)

        # Data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            debug=self.debug
        )

        # Model
        self.model = DSPRNet().to(self.device)

        # Utils
        self.scaler = InverseScaler()
        self.metric = LaplaceMetric()
        self.criterion = MetricAlignedLoss()

        # Optimizer with Differential Learning Rates
        # Group parameters
        backbone_params = list(self.model.encoder.parameters())
        mlp_params = (
            list(self.model.linear_trend.parameters())
            + list(self.model.interaction_mlp.parameters())
            + list(self.model.head.parameters())
        )

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.LR_BACKBONE},
                {"params": mlp_params, "lr": Config.LR_MLP},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        # Cite Lesson 100: Dynamically link Scheduler Horizon to Training Duration
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

        # State
        self.best_score = -float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]",
            leave=False,
        )

        for batch in pbar:
            # Move data to device
            imgs = batch["image"].to(self.device)
            clinical = batch["clinical"].to(self.device)
            targets = batch["target"].to(self.device)

            # Zero grad
            self.optimizer.zero_grad()

            # Forward
            mu, sigma = self.model(imgs, clinical)

            # Loss
            loss = self.criterion(mu, sigma, targets)

            # Backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return running_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        self.metric.reset()

        with torch.no_grad():
            for batch in self.val_loader:
                imgs = batch["image"].to(self.device)
                clinical = batch["clinical"].to(self.device)

                # Raw targets for metric calculation
                raw_targets = batch["fvc_raw"].numpy()

                # Forward
                mu_scaled, sigma_scaled = self.model(imgs, clinical)

                # Unscale predictions
                mu_scaled = mu_scaled.cpu()
                sigma_scaled = sigma_scaled.cpu()

                mu_raw, sigma_raw = self.scaler(mu_scaled, sigma_scaled)

                # Update metric
                self.metric.update(mu_raw, sigma_raw, raw_targets)

        score = self.metric.compute()
        return score

    def fit(self):
        print(f"Starting training on {self.device}...")
        Config.setup()

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_score = self.validate(epoch)

            # Step scheduler
            self.scheduler.step()

            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.5f} | Val Score: {val_score:.10f}"
            )

            # Save best model
            if val_score > self.best_score:
                print(
                    f"Score improved from {self.best_score:.10f} to {val_score:.10f}. Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)

        print(f"Training complete. Best Val Score: {self.best_score:.10f}")

    def predict(self):
        print("Loading best model for inference...")
        if not os.path.exists(Config.BEST_MODEL_PATH):
            print(
                "No checkpoint found. Using current model state (warning: might be suboptimal)."
            )
        else:
            self.model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )

        self.model.eval()
        results = []

        print("Generating predictions on test set...")
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Inference"):
                imgs = batch["image"].to(self.device)
                clinical = batch["clinical"].to(self.device)
                patient_weeks = batch["patient_week"]

                # Forward
                mu_scaled, sigma_scaled = self.model(imgs, clinical)

                # Unscale
                mu_scaled = mu_scaled.cpu()
                sigma_scaled = sigma_scaled.cpu()

                mu_raw, sigma_raw = self.scaler(mu_scaled, sigma_scaled)

                # Post-processing: Clip sigma at 70ml
                sigma_final = torch.clamp(sigma_raw, min=Config.SIGMA_MIN)

                # Collect results
                for pw, fvc, conf in zip(patient_weeks, mu_raw, sigma_final):
                    results.append(
                        {
                            "Patient_Week": pw,
                            "FVC": fvc.item(),
                            "Confidence": conf.item(),
                        }
                    )

        # Save submission
        submission = pd.DataFrame(results)

        # Ensure correct column order
        submission = submission[["Patient_Week", "FVC", "Confidence"]]

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_and_predict():
    """
    Main execution function.
    """
    seed_everything(Config.SEED)

    # Initialize trainer
    trainer = Trainer(debug=Config.DEBUG)

    # Train
    trainer.fit()

    # Predict
    trainer.predict()
