import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    save_checkpoint,
    laplace_log_likelihood,
)
from library.data import get_dataloaders
from library.model import OCPNet


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for OCP-Net.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model = OCPNet().to(self.device)

        # Configure Differential Learning Rates
        # Group 1: Pre-trained Backbone (slower learning)
        backbone_params = list(self.model.image_encoder.parameters())

        # Group 2: New Heads (faster learning)
        head_params = (
            list(self.model.fusion_mlp.parameters())
            + list(self.model.traj_head.parameters())
            + list(self.model.unc_head.parameters())
        )

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.LR_BACKBONE},
                {"params": head_params, "lr": Config.LR_HEADS},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        self.best_score = -float("inf")
        self.patience_counter = 0

    def criterion(self, mu, sigma, target):
        """
        Negative Log Likelihood for Laplace Distribution on normalized data.
        Loss = log(2*sigma) + |target - mu| / sigma
        """
        # Ensure numerical stability for sigma (uncertainty)
        # We use a small floor during training to prevent division by zero or log(0)
        sigma = torch.clamp(sigma, min=Config.MIN_UNCERTAINTY)

        # Calculate NLL
        # Note: We ignore the constant log(2) for optimization purposes
        nll = torch.log(sigma) + torch.abs(target - mu) / sigma
        return nll.mean()

    def train_one_epoch(self, loader):
        self.model.train()
        losses = AverageMeter()

        for batch in loader:
            # Unpack batch: img, tabular, t_rel, target, pid, current_week
            imgs, tab, t_rel, target, _, _ = batch

            imgs = imgs.to(self.device)
            tab = tab.to(self.device)
            t_rel = t_rel.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            mu, sigma = self.model(imgs, tab, t_rel)

            # Compute loss
            loss = self.criterion(mu, sigma, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), imgs.size(0))

        return losses.avg

    def validate(self, loader):
        self.model.eval()
        true_values = []
        pred_values = []
        pred_sigmas = []

        with torch.no_grad():
            for batch in loader:
                imgs, tab, t_rel, target, _, _ = batch

                imgs = imgs.to(self.device)
                tab = tab.to(self.device)
                t_rel = t_rel.to(self.device)

                # Forward pass
                mu_norm, sigma_norm = self.model(imgs, tab, t_rel)

                # Inverse Transformation to original scale (ml)
                # mu_real = mu_norm * std + mean
                # sigma_real = sigma_norm * std
                mu = mu_norm * Config.TARGET_STD + Config.TARGET_MEAN
                sigma = sigma_norm * Config.TARGET_STD

                # Un-normalize target for metric calculation
                target_raw = target * Config.TARGET_STD + Config.TARGET_MEAN

                # Collect results
                true_values.extend(target_raw.cpu().numpy().flatten())
                pred_values.extend(mu.cpu().numpy().flatten())
                pred_sigmas.extend(sigma.cpu().numpy().flatten())

        # Calculate Competition Metric
        score = laplace_log_likelihood(
            np.array(true_values), np.array(pred_values), np.array(pred_sigmas)
        )
        return score

    def fit(self, train_loader, val_loader):
        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch(train_loader)
            val_score = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step()

            print(
                f"Epoch {epoch + 1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Score: {val_score:.6f}"
            )

            # Checkpoint & Early Stopping
            if val_score > self.best_score:
                self.best_score = float(val_score)
                self.patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "best_score": self.best_score,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    is_best=True,
                )
            else:
                self.patience_counter += 1

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    def generate_submission(self, test_loader):
        print("Generating submission...")

        # Load best model weights
        best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if not os.path.exists(best_path):
            print(
                "No checkpoint found! Using current model weights (likely untrained)."
            )
        else:
            checkpoint = torch.load(
                best_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["state_dict"])
            print(
                f"Loaded best model with score: {checkpoint.get('best_score', 'N/A')}"
            )

        self.model.eval()

        # Load sample submission to identify required predictions
        sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

        # Parse Patient and Week from "Patient_Week" (e.g., ID000_12)
        sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
        sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

        results = []

        with torch.no_grad():
            # Iterate over unique patients in the test loader
            # Note: test_loader yields batches, but we process patient-by-patient
            # or batch-by-batch logic below.
            for batch in test_loader:
                imgs, tabs, _, _, pids, base_weeks = batch

                imgs = imgs.to(self.device)
                tabs = tabs.to(self.device)

                # Process each patient in the batch
                for i in range(len(pids)):
                    pid = pids[i]
                    base_week = base_weeks[i].item()

                    # Filter submission requests for this patient
                    patient_sub = sub_df[sub_df["Patient"] == pid]
                    if len(patient_sub) == 0:
                        continue

                    req_weeks = patient_sub["Weeks"].values

                    # Prepare inputs for inference
                    # We repeat the static image/tabular features for each requested week
                    n_samples = len(req_weeks)

                    # (N, 3, H, W)
                    img_rep = imgs[i].unsqueeze(0).repeat(n_samples, 1, 1, 1)
                    # (N, 4)
                    tab_rep = tabs[i].unsqueeze(0).repeat(n_samples, 1)

                    # Calculate relative time for the requested weeks
                    # t_rel = (Target_Week - Baseline_Week) / Scale
                    t_rels = (req_weeks - base_week) / Config.TIME_SCALE
                    t_rels = (
                        torch.tensor(t_rels, dtype=torch.float32)
                        .unsqueeze(1)
                        .to(self.device)
                    )

                    # Predict
                    mu_norm, sigma_norm = self.model(img_rep, tab_rep, t_rels)

                    # Inverse Transform
                    mu = (
                        mu_norm.cpu().numpy().flatten() * Config.TARGET_STD
                        + Config.TARGET_MEAN
                    )
                    sigma = sigma_norm.cpu().numpy().flatten() * Config.TARGET_STD

                    # Apply final clipping for submission as per metric definition
                    sigma = np.maximum(sigma, Config.SUBMISSION_STD_CLIP)

                    # Store results
                    for w, fvc, conf in zip(req_weeks, mu, sigma):
                        results.append(
                            {
                                "Patient_Week": f"{pid}_{w}",
                                "FVC": fvc,
                                "Confidence": conf,
                            }
                        )

        # Create final dataframe
        pred_df = pd.DataFrame(results)

        # Merge with sample submission to ensure correct order and rows
        final_submission = pd.merge(
            sub_df[["Patient_Week"]], pred_df, on="Patient_Week", how="left"
        )

        # Fill missing values (fallback, though logic should cover all)
        final_submission["FVC"] = final_submission["FVC"].fillna(2000)
        final_submission["Confidence"] = final_submission["Confidence"].fillna(100)

        # Save
        final_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Main execution function.
    """
    seed_everything()

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Initialize Trainer
    trainer = Trainer()

    # Train
    trainer.fit(train_loader, val_loader)

    # Generate Submission
    trainer.generate_submission(test_loader)
