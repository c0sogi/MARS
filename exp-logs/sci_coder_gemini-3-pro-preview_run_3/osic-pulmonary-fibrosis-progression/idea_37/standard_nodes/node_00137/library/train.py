import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, get_test_loader
from library.model import DSPRNet


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Continuous Laplace Log Likelihood Loss for optimization.
    L = (sqrt(2) * |y - pred|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.sqrt2 = Config.SQRT2

    def forward(self, mu, sigma, target):
        # inputs are expected to be in standardized space
        # sigma is already processed via softplus in the model

        delta = torch.abs(target - mu)
        term_1 = (self.sqrt2 * delta) / sigma
        term_2 = torch.log(self.sqrt2 * sigma)

        loss = torch.mean(term_1 + term_2)
        return loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = LaplaceLogLikelihoodLoss()

        # Differential Learning Rates
        # Group parameters: Backbone vs Heads
        backbone_params = []
        head_params = []

        # Get IDs of parameters in the backbone
        backbone_ptr = model.backbone
        backbone_ids = set(map(id, backbone_ptr.parameters()))

        for param in model.parameters():
            if not param.requires_grad:
                continue

            if id(param) in backbone_ids:
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

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            imgs = batch["image"].to(self.device)
            clinical = batch["clinical"].to(self.device)
            # Target is (B), need (B, 1) for broadcasting against output
            targets = batch["target"].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            mu, sigma = self.model(imgs, clinical)

            loss = self.criterion(mu, sigma, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        return running_loss / len(self.train_loader.dataset)

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0

        all_trues = []
        all_preds = []
        all_sigmas = []

        with torch.no_grad():
            for batch in self.val_loader:
                imgs = batch["image"].to(self.device)
                clinical = batch["clinical"].to(self.device)
                targets = batch["target"].to(self.device).unsqueeze(1)
                raw_fvcs = batch["raw_fvc"].numpy()

                mu, sigma = self.model(imgs, clinical)

                loss = self.criterion(mu, sigma, targets)
                running_loss += loss.item() * imgs.size(0)

                # Inverse Transform for Metric Calculation
                mu_np = mu.cpu().numpy()
                sigma_np = sigma.cpu().numpy()

                # De-standardize
                mu_orig = mu_np * Config.TARGET_STD + Config.TARGET_MEAN
                sigma_orig = sigma_np * Config.TARGET_STD

                all_trues.extend(raw_fvcs)
                all_preds.extend(mu_orig.flatten())
                all_sigmas.extend(sigma_orig.flatten())

        avg_loss = running_loss / len(self.val_loader.dataset)
        metric_score = calculate_metric(all_trues, all_preds, all_sigmas)

        return avg_loss, metric_score

    def fit(self, epochs, patience=10):
        best_metric = -float("inf")
        patience_counter = 0
        best_epoch = 0

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metric = self.validate_epoch()
            self.scheduler.step()

            # Print full precision as requested
            print(
                f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Metric: {val_metric}"
            )

            # Metric is negative, higher is better
            if val_metric > best_metric:
                best_metric = val_metric
                best_epoch = epoch
                patience_counter = 0

                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model at epoch {epoch}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        print(f"Training finished. Best Metric: {best_metric} at Epoch {best_epoch}")


def generate_submission(model, device):
    """
    Generates submission.csv by predicting FVC for all Patient_Week combinations
    in sample_submission.csv.
    """
    print("Generating submission...")
    model.eval()

    # Load sample submission to get target weeks
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    sub_df = pd.read_csv(sample_sub_path)

    # Pre-load test patient data (static features + image)
    # Batch size 1 ensures we handle one patient at a time
    test_loader = get_test_loader(batch_size=1, num_workers=0)

    patient_data = {}
    with torch.no_grad():
        for batch in test_loader:
            pid = batch["patient_id"][0]
            patient_data[pid] = {
                "image": batch["image"],  # (1, 3, H, W)
                "base_week": batch["base_week"].item(),
                "clinical_base": batch["clinical"],  # (1, 5) where time is 0
            }

    # Parse submission file
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    final_preds = []

    # Iterate by patient to batch predictions
    for pid, group in sub_df.groupby("Patient"):
        if pid not in patient_data:
            continue

        p_data = patient_data[pid]
        weeks = group["Weeks"].values
        n_samples = len(weeks)

        # 1. Expand Image: (N, 3, H, W)
        imgs = p_data["image"].repeat(n_samples, 1, 1, 1).to(device)

        # 2. Construct Clinical Vectors
        # Base: [Base_FVC_Std, Time=0, Age_Std, Sex, Smoke]
        clinicals = p_data["clinical_base"].repeat(n_samples, 1)  # (N, 5)

        # Calculate relative time for each requested week
        base_week = p_data["base_week"]
        rel_times = weeks - base_week
        time_scaled = rel_times * Config.TIME_SCALE

        # Update Time column (index 1)
        clinicals[:, 1] = torch.tensor(time_scaled, dtype=torch.float32)
        clinicals = clinicals.to(device)

        # 3. Inference
        with torch.no_grad():
            mu, sigma = model(imgs, clinicals)

        # 4. Inverse Transform
        mu = mu.cpu().numpy().flatten()
        sigma = sigma.cpu().numpy().flatten()

        mu_orig = mu * Config.TARGET_STD + Config.TARGET_MEAN
        sigma_orig = sigma * Config.TARGET_STD

        # Apply submission clipping logic for sigma
        sigma_orig = np.maximum(sigma_orig, Config.CONFIDENCE_CLIP)

        # 5. Collect Results
        for i, week in enumerate(weeks):
            patient_week = f"{pid}_{week}"
            final_preds.append(
                {
                    "Patient_Week": patient_week,
                    "FVC": mu_orig[i],
                    "Confidence": sigma_orig[i],
                }
            )

    # Create DataFrame and Save
    submission = pd.DataFrame(final_preds)
    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run(epochs=Config.EPOCHS, debug=Config.DEBUG):
    """
    Main entry point for training and submission.
    """
    seed_everything()
    Config.setup()
    device = Config.DEVICE

    # Data
    train_loader, val_loader = get_dataloaders(debug=debug)

    # Model
    model = DSPRNet().to(device)

    # Training
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit(epochs)

    # Load best model for submission
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Submission
    generate_submission(model, device)
