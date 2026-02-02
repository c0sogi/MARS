import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import AttentionFusedDualAxisNet


class LaplaceLoss(nn.Module):
    """
    Computes the negative Modified Laplace Log Likelihood for the parametric model.
    Minimizing this loss is equivalent to maximizing the competition metric.
    """

    def __init__(self):
        super(LaplaceLoss, self).__init__()

    def forward(
        self, alpha, sigma_base, sigma_growth, time_delta, baseline_fvc, target
    ):
        # 1. Calculate Parametric Predictions
        # FVC_pred = Baseline + Alpha * t
        fvc_pred = baseline_fvc + alpha * time_delta

        # Sigma = Sigma_base + Sigma_growth * |t|
        # Note: Softplus is already applied to sigmas in the model output, so they are positive.
        sigma = sigma_base + sigma_growth * torch.abs(time_delta)

        # 2. Apply Metric Constraints
        # Clip confidence at 70ml
        sigma_clipped = torch.clamp(sigma, min=Config.MIN_CONFIDENCE_CLIP)

        # Calculate absolute error
        abs_error = torch.abs(target - fvc_pred)

        # Threshold error at 1000ml
        delta = torch.clamp(abs_error, max=Config.MAX_ERROR_CLIP)

        # 3. Calculate Metric Terms
        # Metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
        # Loss = -Metric
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=target.device))

        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = LaplaceLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
        )

        self.best_val_score = -float("inf")
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        # Use tqdm for progress tracking if interactive, otherwise silent or simple print
        # Given instructions to not print progress bars, we iterate directly
        for batch in self.train_loader:
            # Move data to device
            axial = batch["axial"].to(self.device)
            coronal = batch["coronal"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            target = batch["target"].to(self.device)
            time_delta = batch["time_delta"].to(self.device)
            baseline_fvc = batch["baseline_fvc"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            alpha, sigma_base, sigma_growth = self.model(axial, coronal, tabular)

            # Compute loss
            loss = self.criterion(
                alpha, sigma_base, sigma_growth, time_delta, baseline_fvc, target
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()

        all_true = []
        all_pred = []
        all_sigma = []

        with torch.no_grad():
            for batch in self.val_loader:
                axial = batch["axial"].to(self.device)
                coronal = batch["coronal"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                target = batch["target"].to(self.device)
                time_delta = batch["time_delta"].to(self.device)
                baseline_fvc = batch["baseline_fvc"].to(self.device)

                # Inference
                alpha, sigma_base, sigma_growth = self.model(axial, coronal, tabular)

                # Reconstruct FVC and Sigma
                fvc_pred = baseline_fvc + alpha * time_delta
                sigma = sigma_base + sigma_growth * torch.abs(time_delta)

                all_true.extend(target.cpu().numpy())
                all_pred.extend(fvc_pred.cpu().numpy())
                all_sigma.extend(sigma.cpu().numpy())

        # Calculate official metric
        score = laplace_log_likelihood_metric(
            np.array(all_true), np.array(all_pred), np.array(all_sigma)
        )
        return score

    def fit(self):
        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_score = self.validate()

            # Step scheduler
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss} | "
                f"Val Score: {val_score}"
            )

            # Checkpoint
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with score: {val_score}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Validation Score: {self.best_val_score}")


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    # Load test metadata to align predictions
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if Config.DEBUG:
        test_df = test_df.head(Config.DEBUG_SIZE)

    predictions = []
    confidences = []

    # Ensure order matches test_loader (shuffle=False)
    with torch.no_grad():
        for batch in test_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Predict parameters
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Calculate final FVC and Confidence
            fvc_pred = baseline_fvc + alpha * time_delta
            sigma = sigma_base + sigma_growth * torch.abs(time_delta)

            predictions.extend(fvc_pred.cpu().numpy())
            confidences.extend(sigma.cpu().numpy())

    # Assign to dataframe
    test_df["FVC"] = predictions
    test_df["Confidence"] = confidences

    # Format submission
    submission = test_df[["Patient_Week", "FVC", "Confidence"]].copy()

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())


def run_training(debug=False):
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 3. Model
    device = torch.device(Config.DEVICE)
    model = AttentionFusedDualAxisNet().to(device)

    # 4. Train
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # 5. Inference
    # Load best model
    model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))
    generate_submission(model, test_loader, device)
