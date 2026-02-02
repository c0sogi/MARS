import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.config import Config
from library.utils import seed_everything, AverageMeter, score_function
from library.data import get_dataloaders
from library.model import MACRNet
from library.loss import MetricAlignedLLLoss


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the MACR-Net model.
    """

    def __init__(self, debug=Config.DEBUG):
        self.device = torch.device(Config.DEVICE)
        self.debug = debug

        # Data
        self.train_loader, self.val_loader, self.test_loader, self.stats = (
            get_dataloaders(debug=self.debug)
        )

        # Model
        self.model = MACRNet()
        self.model.to(self.device)

        # Loss
        self.criterion = MetricAlignedLLLoss()

        # Optimizer with Differential Learning Rates
        # Filter parameters for backbone vs heads
        backbone_ids = list(map(id, self.model.visual_stream.backbone.parameters()))
        backbone_params = filter(
            lambda p: id(p) in backbone_ids, self.model.parameters()
        )
        head_params = filter(
            lambda p: id(p) not in backbone_ids, self.model.parameters()
        )

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.LR_BACKBONE},
                {"params": head_params, "lr": Config.LR_HEAD},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        for batch_idx, (imgs, tabs, targets) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            tabs = tabs.to(self.device)
            targets = targets.to(self.device).view(-1, 1)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                preds = self.model(imgs, tabs)
                loss = self.criterion(preds, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.update(loss.item(), imgs.size(0))

        return losses.avg

    def validate(self):
        self.model.eval()
        losses = AverageMeter()
        metrics = AverageMeter()

        with torch.no_grad():
            for imgs, tabs, targets in self.val_loader:
                imgs = imgs.to(self.device)
                tabs = tabs.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                    preds = self.model(imgs, tabs)
                    loss = self.criterion(preds, targets)

                # Calculate Metric
                # 1. Get raw outputs
                mu_pred = preds[:, 0]
                raw_sigma = preds[:, 1]

                # 2. Process Sigma (Softplus)
                sigma_pred = F.softplus(raw_sigma) + 1e-6

                # 3. Inverse Transform for Metric Calculation
                # The metric expects real-world values (ml), not Z-scores
                mu_orig = mu_pred * self.stats["FVC_std"] + self.stats["FVC_mean"]
                sigma_orig = sigma_pred * self.stats["FVC_std"]
                targets_orig = (
                    targets.squeeze() * self.stats["FVC_std"] + self.stats["FVC_mean"]
                )

                # 4. Compute Metric
                score = score_function(targets_orig, mu_orig, sigma_orig)

                losses.update(loss.item(), imgs.size(0))
                metrics.update(score, imgs.size(0))

        return losses.avg, metrics.avg

    def fit(self, epochs=Config.EPOCHS, patience=10):
        print(f"Starting training for {epochs} epochs on {self.device}...")
        best_metric = -float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_metric = self.validate()

            self.scheduler.step()

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Metric: {val_metric}"
            )

            # Save Best Model
            # Metric is negative and higher is better
            if val_metric > best_metric:
                best_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"  >>> New Best Model Saved! Metric: {best_metric}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

        print(f"Training complete. Best Validation Metric: {best_metric}")

    def predict_and_submit(self):
        print("Generating submission...")

        # Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            print(
                "No checkpoint found! Using current model state (likely untrained or interrupted)."
            )
        else:
            self.model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )
            print("Loaded best model checkpoint.")

        self.model.eval()

        results = []

        # Re-construct the test dataframe logic to map predictions back to Patient_Week
        # We need to iterate the loader which corresponds to the sample_submission rows
        # The test_loader is built from sample_submission.csv in get_dataloaders

        # We need the original sample_submission to map indices correctly or just append
        # Since DataLoader preserves order (shuffle=False), we can map directly.

        predictions_mu = []
        predictions_sigma = []

        with torch.no_grad():
            for imgs, tabs in self.test_loader:
                imgs = imgs.to(self.device)
                tabs = tabs.to(self.device)

                with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                    preds = self.model(imgs, tabs)

                mu_pred = preds[:, 0]
                raw_sigma = preds[:, 1]

                sigma_pred = F.softplus(raw_sigma) + 1e-6

                predictions_mu.extend(mu_pred.cpu().numpy())
                predictions_sigma.extend(sigma_pred.cpu().numpy())

        # Inverse Transform
        predictions_mu = np.array(predictions_mu)
        predictions_sigma = np.array(predictions_sigma)

        final_mu = predictions_mu * self.stats["FVC_std"] + self.stats["FVC_mean"]
        final_sigma = predictions_sigma * self.stats["FVC_std"]

        # Apply Hard Clip for Submission (as per logic: max(sigma, 70))
        final_sigma = np.maximum(final_sigma, 70.0)

        # Load sample submission to fill in values
        sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

        # Ensure lengths match
        if len(sub_df) != len(final_mu):
            print(
                f"Warning: Submission length mismatch! DF: {len(sub_df)}, Preds: {len(final_mu)}"
            )

        sub_df["FVC"] = final_mu
        sub_df["Confidence"] = final_sigma

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    seed_everything(Config.SEED)

    trainer = Trainer()
    trainer.fit()
    trainer.predict_and_submit()
