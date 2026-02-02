import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import PCDSNet
from library.utils import AverageMeter, metric_laplace_log_likelihood


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Continuous Laplace Log Likelihood Loss for training.
    Formula: L = (sqrt(2) * |y - pred|) / sigma + ln(sqrt(2) * sigma)

    Note: This is the Negative Log Likelihood (NLL) to be minimized.
    It does NOT include the metric's clipping (70ml) or thresholding (1000ml)
    to ensure smooth gradients for optimization.
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()

    def forward(self, mu, sigma, target):
        # mu, sigma, target are expected to be on the same device
        # target is the normalized ground truth

        # Constants
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=mu.device))

        # Absolute error
        delta = torch.abs(target - mu)

        # Calculate NLL
        # sigma is guaranteed positive via softplus in the model
        loss = (sqrt_2 * delta) / sigma + torch.log(sqrt_2 * sigma)

        return torch.mean(loss)


class Runner:
    def __init__(
        self, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, device=Config.DEVICE
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        # Create directories
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Data
        self.train_loader, self.val_loader, self.test_loader, self.stats = (
            get_dataloaders(batch_size=self.batch_size, num_workers=Config.NUM_WORKERS)
        )

        # Model
        self.model = PCDSNet()
        self.model.to(self.device)

        # Optimizer with Differential Learning Rates
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.LR_BACKBONE},
                {"params": head_params, "lr": Config.LR_HEAD},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        # Loss
        self.criterion = LaplaceLogLikelihoodLoss()

        # Tracking
        self.best_metric = -float("inf")
        self.best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        loss_meter = AverageMeter()

        # Loop over batches
        # Using tqdm for visual feedback in logs if needed, but keeping it minimal
        for img, clinical, target in self.train_loader:
            img = img.to(self.device)
            clinical = clinical.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            mu, sigma = self.model(img, clinical)

            loss = self.criterion(mu, sigma, target)
            loss.backward()

            self.optimizer.step()

            loss_meter.update(loss.item(), img.size(0))

        return loss_meter.avg

    def validate(self):
        self.model.eval()
        metric_meter = AverageMeter()

        fvc_mean = self.stats["fvc_mean"]
        fvc_std = self.stats["fvc_std"]

        with torch.no_grad():
            for img, clinical, target in self.val_loader:
                img = img.to(self.device)
                clinical = clinical.to(self.device)
                target = target.to(self.device)  # Normalized target

                mu_norm, sigma_norm = self.model(img, clinical)

                # Denormalize for metric calculation
                # mu_abs = mu_norm * std + mean
                # sigma_abs = sigma_norm * std
                # target_abs = target_norm * std + mean

                mu_abs = mu_norm * fvc_std + fvc_mean
                sigma_abs = sigma_norm * fvc_std
                target_abs = target * fvc_std + fvc_mean

                # Calculate metric using the official competition logic
                score = metric_laplace_log_likelihood(target_abs, mu_abs, sigma_abs)
                metric_meter.update(score, img.size(0))

        return metric_meter.avg

    def run(self):
        print(f"Starting training for {self.epochs} epochs on {self.device}...")

        patience = 10
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_metric = self.validate()

            self.scheduler.step()

            print(
                f"Epoch {epoch}/{self.epochs} | Train Loss: {train_loss:.6f} | Val Metric: {val_metric:.10f}"
            )

            # Checkpoint
            if val_metric > self.best_metric:
                self.best_metric = val_metric
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  >>> New Best Model Saved! Metric: {self.best_metric:.10f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

        print(f"Training complete. Best Validation Metric: {self.best_metric:.10f}")

    def generate_submission(self):
        print("Generating submission...")

        # Load best model
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()

        results = []
        fvc_mean = self.stats["fvc_mean"]
        fvc_std = self.stats["fvc_std"]

        with torch.no_grad():
            for img, clinical, patient_week_ids in self.test_loader:
                img = img.to(self.device)
                clinical = clinical.to(self.device)

                mu_norm, sigma_norm = self.model(img, clinical)

                # Denormalize
                mu_abs = mu_norm * fvc_std + fvc_mean
                sigma_abs = sigma_norm * fvc_std

                # Move to CPU
                mu_abs = mu_abs.cpu().numpy()
                sigma_abs = sigma_abs.cpu().numpy()

                # Process batch
                for i, pw_id in enumerate(patient_week_ids):
                    pred_fvc = mu_abs[i]
                    pred_sigma = sigma_abs[i]

                    # Apply final submission constraint: sigma >= 70
                    pred_sigma = max(pred_sigma, 70)

                    results.append(
                        {
                            "Patient_Week": pw_id,
                            "FVC": pred_fvc,
                            "Confidence": pred_sigma,
                        }
                    )

        # Create DataFrame
        sub_df = pd.DataFrame(results)

        # Ensure column order
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

        # Save
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")


def main():
    seed_everything(Config.SEED)

    runner = Runner()
    runner.run()
    runner.generate_submission()


# Note: The main execution block is omitted as per instructions.
# The class/functions above are ready to be imported or run.
