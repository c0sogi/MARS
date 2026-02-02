import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    seed_everything,
    laplace_log_likelihood_metric,
    inverse_transform_predictions,
)
from library.data import get_dataloaders
from library.model import DSPRNet


class MetricAlignedLLLoss(nn.Module):
    """
    Loss function aligned with the competition metric: Modified Laplace Log Likelihood.
    Cite Lesson 66: Derive loss directly from metric formula (including constants).
    Cite Lesson 138: Avoid hard clipping (min/max/clamp) in the loss function to prevent gradient vanishing.
    """

    def __init__(self):
        super().__init__()
        self.target_std = Config.TARGET_STD
        self.target_mean = Config.TARGET_MEAN
        # Precompute sqrt(2)
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 2) Tensor. Col 0: Normalized Mu. Col 1: Raw Sigma score.
            target: (B,) Tensor. Normalized FVC.
        """
        mu_norm = pred[:, 0]
        raw_sigma_norm = pred[:, 1]

        # Enforce positivity for sigma using Softplus
        sigma_norm = F.softplus(raw_sigma_norm)

        # Inverse transform to absolute scale (ml)
        mu_abs = mu_norm * self.target_std + self.target_mean
        sigma_abs = sigma_norm * self.target_std
        target_abs = target * self.target_std + self.target_mean

        # Calculate Delta (No clipping for gradients)
        delta = torch.abs(target_abs - mu_abs)

        # Calculate Sigma (No clipping for gradients, just stability epsilon)
        sigma_safe = sigma_abs + 1e-6

        # Metric formula: - (sqrt(2) * delta / sigma) - ln(sqrt(2) * sigma)
        # Loss = -Metric
        term1 = (self.sqrt_2 * delta) / sigma_safe
        term2 = torch.log(self.sqrt_2 * sigma_safe)

        loss = term1 + term2
        return torch.mean(loss)


def run_training(debug=Config.DEBUG, epochs=Config.EPOCHS):
    """
    Main training loop for DSPR-Net.
    """
    # 1. Setup
    Config.DEBUG = debug
    Config.EPOCHS = epochs
    seed_everything(Config.SEED)
    Config.setup()  # Ensure directories exist

    print(
        f"Initializing training. Debug={Config.DEBUG}, Epochs={Config.EPOCHS}, Device={Config.DEVICE}"
    )

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model
    model = DSPRNet().to(Config.DEVICE)

    # 4. Optimizer with Differential Learning Rates
    # Separate backbone parameters from the rest (heads, clinical stream, etc.)
    backbone_ids = list(map(id, model.visual_stream.backbone.parameters()))

    backbone_params = []
    head_params = []

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in backbone_ids:
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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = MetricAlignedLLLoss().to(Config.DEVICE)

    # 5. Training Loop
    best_score = -float("inf")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_losses = []

        for batch in train_loader:
            imgs, tabs, targets = batch
            imgs = imgs.to(Config.DEVICE)
            tabs = tabs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            optimizer.zero_grad()

            # Forward pass (Cite Lesson 34: No auxiliary loss)
            final_out = model(imgs, tabs)

            # Loss calculation
            loss = criterion(final_out, targets)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation Step
        model.eval()
        val_preds_mu = []
        val_preds_sigma = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                imgs, tabs, targets = batch
                imgs = imgs.to(Config.DEVICE)
                tabs = tabs.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)

                final_out = model(imgs, tabs)

                # Process outputs
                mu_norm = final_out[:, 0]
                raw_sigma_norm = final_out[:, 1]
                sigma_norm = F.softplus(raw_sigma_norm)

                # Inverse transform for metric calculation
                mu_abs, sigma_abs = inverse_transform_predictions(mu_norm, sigma_norm)
                target_abs = targets * Config.TARGET_STD + Config.TARGET_MEAN

                val_preds_mu.extend(mu_abs.cpu().numpy())
                val_preds_sigma.extend(sigma_abs.cpu().numpy())
                val_targets.extend(target_abs.cpu().numpy())

        # Calculate Competition Metric
        val_score = laplace_log_likelihood_metric(
            np.array(val_targets), np.array(val_preds_mu), np.array(val_preds_sigma)
        )

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {np.mean(train_losses):.6f} | Val Score: {val_score:.6f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training complete. Best Validation Score: {best_score:.6f}")

    # 6. Submission Generation
    print("Generating submission...")
    generate_submission(model, test_loader)


def generate_submission(model, test_loader):
    """
    Loads best model, predicts on test set, and saves submission file.
    """
    # Load best weights
    model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    test_preds_mu = []
    test_preds_sigma = []

    with torch.no_grad():
        for batch in test_loader:
            imgs, tabs, _ = batch  # Ignore dummy targets
            imgs = imgs.to(Config.DEVICE)
            tabs = tabs.to(Config.DEVICE)

            final_out = model(imgs, tabs)

            mu_norm = final_out[:, 0]
            raw_sigma_norm = final_out[:, 1]
            sigma_norm = F.softplus(raw_sigma_norm)

            mu_abs, sigma_abs = inverse_transform_predictions(mu_norm, sigma_norm)

            test_preds_mu.extend(mu_abs.cpu().numpy())
            test_preds_sigma.extend(sigma_abs.cpu().numpy())

    # Prepare DataFrame
    # Access the dataframe from the dataset to get Patient_Week IDs
    test_df = test_loader.dataset.df
    submission = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": test_preds_mu,
            "Confidence": test_preds_sigma,
        }
    )

    # Apply final clipping for submission as per rules
    submission["Confidence"] = submission["Confidence"].apply(
        lambda x: max(x, Config.SIGMA_CLIP)
    )

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
