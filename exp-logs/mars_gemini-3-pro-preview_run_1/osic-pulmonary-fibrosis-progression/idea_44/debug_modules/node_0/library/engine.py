import os
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.model import SLADAN
from library.data import get_dataloaders
from library.utils import AverageMeter, score_function


class LaplaceLoss(nn.Module):
    """
    Differentiable Laplace Log Likelihood Loss with Straight-Through Estimator (STE)
    for clipping bounds.

    Metric formulation:
        delta = min(|true - pred|, 1000)
        sigma_clipped = max(sigma, 70)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    We minimize Loss = -Metric.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sqrt2 = math.sqrt(2)

    def forward(
        self, alpha, sigma_base, sigma_growth, target, week, base_fvc, base_week
    ):
        # 1. Reconstruct Predictions based on Anchored Trajectory Logic
        # FVC_pred = Base_FVC + alpha * (Week - Base_Week)
        dt = week - base_week
        pred_fvc = base_fvc + alpha * dt

        # Sigma_pred = Sigma_base + Sigma_growth * |Week - Base_Week|
        pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

        # 2. Calculate Error (Delta)
        abs_err = torch.abs(target - pred_fvc)

        # 3. Apply Clipping with Straight-Through Estimator (STE)
        # We want the value to be clipped, but the gradient to flow through
        # to encourage the model to reduce error/increase sigma even if outside bounds.

        # Delta: max 1000
        # Value: min(|err|, 1000)
        delta_clipped = torch.clamp(abs_err, max=self.cfg.MAX_ERROR)
        # STE: delta = clipped + (raw - raw).detach() -> Gradient flows to raw
        # However, for Delta, if error > 1000, we DO want gradient to pull it down.
        # Standard clamp has grad=0. We add the residual.
        delta = delta_clipped + (abs_err - abs_err.detach())

        # Sigma: min 70
        sigma_clamped = torch.clamp(pred_sigma, min=self.cfg.CONFIDENCE_CLIP)
        # STE: If sigma < 70, we use 70 for calc, but pass gradient to sigma
        sigma_clipped = sigma_clamped + (pred_sigma - pred_sigma.detach())

        # 4. Calculate Negative Metric (Loss)
        # Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
        term1 = (self.sqrt2 * delta) / sigma_clipped
        term2 = torch.log(self.sqrt2 * sigma_clipped)

        loss = torch.mean(term1 + term2)
        return loss


class Engine:
    def __init__(self, debug=False, epochs=50):
        # Initialize Config with overrides
        self.cfg = Config(debug=debug, epochs=epochs)
        self.device = self.cfg.DEVICE

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(self.cfg)

        # Build Baseline Lookups (Patient -> {base_fvc, base_week})
        # Required because DataLoader does not return baseline FVC for training rows
        self.train_lookup = self._build_baseline_lookup(
            self.cfg.TRAIN_CSV, is_test=False
        )
        self.val_lookup = self._build_baseline_lookup(self.cfg.VAL_CSV, is_test=False)
        self.test_lookup = self._build_baseline_lookup(self.cfg.TEST_CSV, is_test=True)

        # Model
        self.model = SLADAN(self.cfg).to(self.device)

        # Optimization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.EPOCHS
        )

        # Loss
        self.criterion = LaplaceLoss(self.cfg)

        # Tracking
        self.best_score = -float("inf")

    def _build_baseline_lookup(self, csv_path, is_test=False):
        """
        Builds a dictionary mapping Patient ID to Baseline FVC and Week.
        """
        df = pd.read_csv(csv_path)
        lookup = {}

        if is_test:
            # Test CSV already has Baseline columns
            # Drop duplicates to get unique patient info
            unique_pats = df[
                ["Patient", "Baseline_FVC", "Baseline_Week"]
            ].drop_duplicates()
            for _, row in unique_pats.iterrows():
                lookup[row["Patient"]] = {
                    "base_fvc": float(row["Baseline_FVC"]),
                    "base_week": float(row["Baseline_Week"]),
                }
        else:
            # Train/Val CSV: Find row with min(Weeks) for each patient
            for pid in df["Patient"].unique():
                p_data = df[df["Patient"] == pid]
                base_idx = p_data["Weeks"].idxmin()
                base_row = p_data.loc[base_idx]
                lookup[pid] = {
                    "base_fvc": float(base_row["FVC"]),
                    "base_week": float(base_row["Weeks"]),
                }
        return lookup

    def _get_baseline_batch(self, patient_ids):
        """
        Retrieves baseline FVC and Week for a batch of patient IDs.
        Checks train, val, and test lookups.
        """
        base_fvcs = []
        base_weeks = []

        for pid in patient_ids:
            # Check lookups in order
            if pid in self.train_lookup:
                data = self.train_lookup[pid]
            elif pid in self.val_lookup:
                data = self.val_lookup[pid]
            elif pid in self.test_lookup:
                data = self.test_lookup[pid]
            else:
                # Fallback (should not happen in valid flow)
                data = {"base_fvc": 2000.0, "base_week": 0.0}

            base_fvcs.append(data["base_fvc"])
            base_weeks.append(data["base_week"])

        return (
            torch.tensor(base_fvcs, dtype=torch.float32, device=self.device),
            torch.tensor(base_weeks, dtype=torch.float32, device=self.device),
        )

    def train_one_epoch(self):
        self.model.train()
        loss_meter = AverageMeter()

        for batch in self.train_loader:
            # Move data to device
            img_ax = batch["img_axial"].to(self.device)
            img_cor = batch["img_coronal"].to(self.device)
            tab = batch["tabular"].to(self.device)
            target = batch["target"].to(self.device)
            week = batch["week"].to(self.device)
            pids = batch["patient_id"]

            # Retrieve baseline info
            base_fvc, base_week = self._get_baseline_batch(pids)

            self.optimizer.zero_grad()

            # Forward Pass
            # Output: [alpha, sigma_base, sigma_growth]
            preds = self.model(img_ax, img_cor, tab)
            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Loss Calculation
            loss = self.criterion(
                alpha, sigma_base, sigma_growth, target, week, base_fvc, base_week
            )

            # Backward Pass
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), img_ax.size(0))

        return loss_meter.avg

    def evaluate(self):
        self.model.eval()
        score_meter = AverageMeter()

        with torch.no_grad():
            for batch in self.val_loader:
                img_ax = batch["img_axial"].to(self.device)
                img_cor = batch["img_coronal"].to(self.device)
                tab = batch["tabular"].to(self.device)
                target = batch["target"].to(self.device)
                week = batch["week"].to(self.device)
                pids = batch["patient_id"]

                base_fvc, base_week = self._get_baseline_batch(pids)

                preds = self.model(img_ax, img_cor, tab)
                alpha = preds[:, 0]
                sigma_base = preds[:, 1]
                sigma_growth = preds[:, 2]

                # Reconstruct Predictions
                dt = week - base_week
                pred_fvc = base_fvc + alpha * dt
                pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

                # Calculate Metric
                # score_function handles CPU conversion and clipping logic internally
                score = score_function(target, pred_fvc, pred_sigma)
                score_meter.update(score, img_ax.size(0))

        return score_meter.avg

    def fit(self):
        print(f"Starting training on device: {self.device}")
        patience_counter = 0

        for epoch in range(self.cfg.EPOCHS):
            train_loss = self.train_one_epoch()
            val_score = self.evaluate()

            # Step Scheduler
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.cfg.EPOCHS} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.6f}"
            )

            # Checkpoint & Early Stopping
            if val_score > self.best_score:
                self.best_score = val_score
                patience_counter = 0
                save_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Validation Score: {self.best_score:.6f}")

    def generate_submission(self):
        print("Generating submission...")
        # Load best model weights
        best_path = os.path.join(self.cfg.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        else:
            print("Warning: Best model not found. Using current weights.")

        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in self.test_loader:
                img_ax = batch["img_axial"].to(self.device)
                img_cor = batch["img_coronal"].to(self.device)
                tab = batch["tabular"].to(self.device)
                week = batch["week"].to(self.device)  # This is Predict_Week
                pids = batch["patient_id"]

                base_fvc, base_week = self._get_baseline_batch(pids)

                preds = self.model(img_ax, img_cor, tab)
                alpha = preds[:, 0]
                sigma_base = preds[:, 1]
                sigma_growth = preds[:, 2]

                # Calculate Predictions
                dt = week - base_week
                pred_fvc = base_fvc + alpha * dt
                pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

                # Move to CPU for formatting
                pred_fvc_np = pred_fvc.cpu().numpy()
                pred_sigma_np = pred_sigma.cpu().numpy()
                weeks_np = week.cpu().numpy()

                for i, pid in enumerate(pids):
                    # Format Patient_Week ID (e.g., ID00007637202177411956430_12)
                    wk = int(weeks_np[i])
                    patient_week = f"{pid}_{wk}"

                    results.append(
                        {
                            "Patient_Week": patient_week,
                            "FVC": pred_fvc_np[i],
                            "Confidence": pred_sigma_np[i],
                        }
                    )

        # Save to CSV
        df = pd.DataFrame(results)
        # Ensure correct column order
        df = df[["Patient_Week", "FVC", "Confidence"]]
        df.to_csv(self.cfg.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.cfg.SUBMISSION_PATH}")
