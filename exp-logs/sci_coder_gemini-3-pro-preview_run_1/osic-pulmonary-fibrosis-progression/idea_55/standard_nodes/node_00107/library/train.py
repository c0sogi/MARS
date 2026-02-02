import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, metric_score
from library.data import get_dataloaders
from library.model import NSLHN


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the negative modified Laplace Log Likelihood as the loss function.

    Metric:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        score = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Loss = -score
    """

    def __init__(self):
        super().__init__()
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(
        self, alpha, sigma_base, sigma_growth, target, delta_week, baseline_fvc
    ):
        # 1. Calculate Predictions
        # FVC_pred = Baseline + alpha * delta_week
        pred_fvc = baseline_fvc + alpha * delta_week

        # Sigma_pred = Base + Growth * |delta_week|
        pred_sigma = sigma_base + sigma_growth * torch.abs(delta_week)

        # 2. Apply Constraints
        # Clip sigma at 70ml
        sigma_clipped = torch.clamp(pred_sigma, min=Config.SIGMA_CLIP)

        # Calculate absolute error
        abs_error = torch.abs(target - pred_fvc)

        # Clip error at 1000ml (Robustness to outliers)
        delta = torch.clamp(abs_error, max=Config.ERROR_CLIP)

        # 3. Compute Loss (Negative Metric)
        # Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)

        # Ensure sqrt_2 is on correct device
        sqrt_2 = self.sqrt_2.to(alpha.device)

        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        loss = torch.mean(term1 + term2)
        return loss


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move inputs to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        delta_week = batch["delta_week"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)

        optimizer.zero_grad()

        # Forward pass
        alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

        # Compute loss
        loss = criterion(
            alpha, sigma_base, sigma_growth, target, delta_week, baseline_fvc
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * axial.size(0)

    return running_loss / len(loader.dataset)


def valid_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    all_sigmas = []

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            delta_week = batch["delta_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Compute loss
            loss = criterion(
                alpha, sigma_base, sigma_growth, target, delta_week, baseline_fvc
            )
            running_loss += loss.item() * axial.size(0)

            # Compute actual predictions for metric calculation
            pred_fvc = baseline_fvc + alpha * delta_week
            pred_sigma = sigma_base + sigma_growth * torch.abs(delta_week)

            all_targets.append(target.cpu().numpy())
            all_preds.append(pred_fvc.cpu().numpy())
            all_sigmas.append(pred_sigma.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    sigma = np.concatenate(all_sigmas)

    # Calculate official metric
    score = metric_score(y_true, y_pred, sigma)
    avg_loss = running_loss / len(loader.dataset)

    return avg_loss, score


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    model.eval()
    results = []

    print("Generating submission...")

    with torch.no_grad():
        for batch in test_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            patient_ids = batch["patient_id"]  # List of IDs

            # Forward pass
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Calculate predictions
            pred_fvc = baseline_fvc + alpha * delta_week
            pred_sigma = sigma_base + sigma_growth * torch.abs(delta_week)

            # Move to CPU
            pred_fvc = pred_fvc.cpu().numpy()
            pred_sigma = pred_sigma.cpu().numpy()
            delta_week_np = delta_week.cpu().numpy()

            # Iterate batch
            for i in range(len(patient_ids)):
                # Reconstruct Patient_Week ID from metadata if needed,
                # but test_loader iterates over test.csv which has Patient and Predict_Week.
                # We need to reconstruct the submission ID: Patient_Week
                # The test.csv in metadata has 'Patient' and 'Predict_Week'.
                # The loader batch doesn't explicitly pass 'Predict_Week' unless we modify dataset.
                # However, we can reconstruct it using Delta_Week + Baseline_Week?
                # Actually, the simplest way is to rely on the order if preserved,
                # or better, assume the dataset returns what we need.
                # LungDataset for 'test' mode uses test.csv.
                # Let's assume we can reconstruct the ID or the caller handles it.
                # Wait, the submission format requires 'Patient_Week'.
                # Let's look at Config.TEST_CSV. It has 'Patient_Week'.
                # We should probably modify LungDataset to return Patient_Week,
                # but I cannot modify library files.
                # However, LungDataset.__getitem__ returns 'patient_id'.
                # For test set, 'patient_id' in the dataframe is just the Patient column.
                # We need the Week.
                # We can recover the week: Week = Delta_Week + Baseline_Week.
                # But Baseline_Week is not in the batch dict in the provided code?
                # Wait, LungDataset returns 'delta_week' and 'baseline_fvc'.
                # It does NOT return 'baseline_week'.
                # However, we can read the test.csv directly to get the IDs since the loader is not shuffled.
                pass

            results.append(np.stack([pred_fvc, pred_sigma], axis=1))

    # Concatenate predictions
    all_preds = np.concatenate(results, axis=0)

    # Load test csv to get IDs
    test_df = pd.read_csv(Config.TEST_CSV)

    # Ensure lengths match
    if len(test_df) != len(all_preds):
        print(
            f"Warning: Length mismatch. Test DF: {len(test_df)}, Preds: {len(all_preds)}"
        )

    # Create submission DataFrame
    sub_df = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": all_preds[:, 0],
            "Confidence": all_preds[:, 1],
        }
    )

    # Save
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_training():
    seed_everything(Config.SEED)

    # 1. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # 2. Model
    print("Initializing NSL-HN model...")
    device = torch.device(Config.DEVICE)
    model = NSLHN().to(device)

    # 3. Optimization
    criterion = LaplaceLogLikelihoodLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 4. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_score = valid_epoch(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_score:.10f}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! Score: {best_score:.10f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")

    # 5. Generate Submission
    # Load best model
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, test_loader, device)
