import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from library.config import Config
from library.data import get_dataloaders
from library.model import AttentionFusedDualAxisNet
from library.utils import seed_everything, laplace_log_likelihood, average_weights


class LaplaceLoss(nn.Module):
    """
    Differentiable implementation of the Modified Laplace Log Likelihood Loss.
    Loss = -Metric (since we want to maximize the metric).
    """

    def __init__(self):
        super().__init__()
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))
        self.max_error = Config.MAX_ERROR
        self.sigma_clip = Config.SIGMA_CLIP

    def forward(self, fvc_pred, sigma_pred, fvc_true):
        # Clip sigma to avoid division by zero or extremely small numbers
        sigma_clipped = torch.clamp(sigma_pred, min=self.sigma_clip)

        # Calculate absolute error
        abs_error = torch.abs(fvc_true - fvc_pred)

        # Clip error to avoid excessive penalization for outliers
        delta = torch.clamp(abs_error, max=self.max_error)

        # Calculate metric terms
        # metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        term1 = (self.sqrt_2.to(delta.device) * delta) / sigma_clipped
        term2 = torch.log(self.sqrt_2.to(delta.device) * sigma_clipped)

        # The metric is negative, so we minimize the negative of the metric
        # Loss = - (Metric) = term1 + term2
        loss = torch.mean(term1 + term2)

        return loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move inputs to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)
        relative_week = batch["relative_week"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)
        fvc_target = batch["fvc_target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        fvc_pred, sigma_pred = model(
            tabular, img_ax, img_cor, relative_week, baseline_fvc
        )

        # Calculate loss
        loss = criterion(fvc_pred, sigma_pred, fvc_target)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_ax.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    model.eval()

    all_fvc_true = []
    all_fvc_pred = []
    all_sigma_pred = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            relative_week = batch["relative_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            fvc_target = batch["fvc_target"].to(device)

            fvc_pred, sigma_pred = model(
                tabular, img_ax, img_cor, relative_week, baseline_fvc
            )

            all_fvc_true.append(fvc_target.cpu().numpy())
            all_fvc_pred.append(fvc_pred.cpu().numpy())
            all_sigma_pred.append(sigma_pred.cpu().numpy())

    y_true = np.concatenate(all_fvc_true)
    y_pred = np.concatenate(all_fvc_pred)
    sigma = np.concatenate(all_sigma_pred)

    score = laplace_log_likelihood(y_true, y_pred, sigma)
    return score


def run_training(debug=False, epochs=None):
    # Setup
    Config.setup_directories()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Overwrite epochs if provided (e.g. for debugging)
    num_epochs = epochs if epochs is not None else Config.EPOCHS
    if debug:
        num_epochs = 2
        print("Debug mode: Training for 2 epochs.")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # Model
    model = AttentionFusedDualAxisNet().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss
    criterion = LaplaceLoss()

    # Tracking
    best_score = -float("inf")
    patience_counter = 0

    # Store (score, epoch, path) for top K checkpoints
    top_k_checkpoints = []

    print(f"Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Score: {val_score}"
        )

        # Checkpoint Logic (Save Top 3)
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

        # Update top K list
        top_k_checkpoints.append((val_score, epoch, ckpt_path))
        # Sort by score descending (higher is better)
        top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Keep only top 3
        if len(top_k_checkpoints) > 3:
            # Remove the worst one from disk to save space
            to_remove = top_k_checkpoints.pop()
            if os.path.exists(to_remove[2]):
                os.remove(to_remove[2])

        # Early Stopping Logic based on best score found so far
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # --- SWA / Model Averaging ---
    if Config.USE_SWA:
        print("\nPerforming SWA on top checkpoints:")
        best_paths = [x[2] for x in top_k_checkpoints]
        for p in best_paths:
            print(f" - {p}")

        # Load and average weights
        model = average_weights(model, best_paths)
    else:
        # If not using SWA, ensure we load the single best model found
        # The 'model' variable currently holds the state at the last epoch.
        # We want the best score model.
        # top_k_checkpoints is sorted descending by score.
        best_ckpt = top_k_checkpoints[0][2]
        print(f"\nLoading best checkpoint from: {best_ckpt}")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Save Final Best Model
    final_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Best model saved to {final_model_path}")

    # --- Inference on Test Set ---
    print("\nGenerating submission...")
    model.eval()
    submission_rows = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            relative_week = batch["relative_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            patient_ids = batch["patient_id"]

            # Predict
            fvc_pred, sigma_pred = model(
                tabular, img_ax, img_cor, relative_week, baseline_fvc
            )

            # Move to CPU
            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()
            relative_week = relative_week.cpu().numpy()

            # The test loader iterates over rows in test.csv.
            # We need to reconstruct Patient_Week ID.
            # However, the batch doesn't directly give us the target week absolute value easily
            # unless we passed it through. But we can reconstruct or just use the order if preserved.
            # Better approach: The test_ds returns 'relative_week'.
            # We need the absolute week for the ID?
            # Actually, test.csv has 'Patient_Week' column.
            # But the Dataset __getitem__ doesn't return it.
            # We can reconstruct it: Patient + "_" + (Baseline_Week + Relative_Week).
            # Let's rely on the fact that test_loader is sequential and matches test.csv rows.

            # Wait, we can't easily rely on order if shuffle is off (it is off).
            # But let's look at test.csv structure in metadata.
            # It has Patient_Week.
            pass

            # Collect results
            for i in range(len(patient_ids)):
                # We need to map back to the specific row.
                # Since we iterate sequentially over test.csv (loader shuffle=False),
                # we can just collect predictions and assign them to the dataframe later.
                submission_rows.append(
                    {"FVC": fvc_pred[i], "Confidence": sigma_pred[i]}
                )

    # Load test csv to assign predictions
    test_df = pd.read_csv(Config.TEST_CSV)

    # Ensure lengths match
    if len(submission_rows) != len(test_df):
        print(
            f"Warning: Prediction count {len(submission_rows)} != Test DF length {len(test_df)}"
        )

    pred_df = pd.DataFrame(submission_rows)

    # Create submission DataFrame
    submission = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": pred_df["FVC"],
            "Confidence": pred_df["Confidence"],
        }
    )

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
