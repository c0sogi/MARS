import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config, set_seed
from library.data_loader import get_dataloaders, get_test_loader
from library.network import ADGN_Model, CosineSimilarityLoss
from library.utils import cartesian_to_spherical, angular_dist_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    for X, priors, y, _ in loader:
        X = X.to(device, non_blocking=True)
        priors = priors.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        preds = model(X, priors)
        loss = criterion(preds, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples if num_samples > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and the competition metric (mean angular error).
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, priors, y, _ in loader:
            X = X.to(device, non_blocking=True)
            priors = priors.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            preds = model(X, priors)
            loss = criterion(preds, y)

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

    # Compute Competition Metric
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Convert Cartesian predictions back to Spherical for metric calculation
        pred_az, pred_zen = cartesian_to_spherical(
            all_preds[:, 0], all_preds[:, 1], all_preds[:, 2]
        )
        y_pred_spherical = np.stack([pred_az, pred_zen], axis=1)

        metric = angular_dist_score(all_targets, y_pred_spherical)
    else:
        metric = 0.0

    return avg_loss, metric


class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Initialize Model
        self.model = ADGN_Model().to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

        # Loss Function
        self.criterion = CosineSimilarityLoss()

    def fit(self):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on {self.device}...")

        # Load Data
        train_loader, val_loader = get_dataloaders()

        best_metric = float("inf")
        best_epoch = -1
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            # Training Step
            train_loss = train_one_epoch(
                self.model, train_loader, self.optimizer, self.criterion, self.device
            )

            # Validation Step
            val_loss, val_metric = validate(
                self.model, val_loader, self.criterion, self.device
            )

            # Update Scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Metric: {val_metric} | "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpointing
            if val_metric < best_metric:
                best_metric = val_metric
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
                print("New best model saved.")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after epoch {epoch+1}.")
                break

        print(f"Training complete. Best Metric: {best_metric} at Epoch {best_epoch+1}")

    def predict_and_submit(self):
        """
        Loads the best model, performs inference on the test set, and saves the submission file.
        """
        print("Loading best model for inference...")
        if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
            print(
                f"Checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}. Cannot generate submission."
            )
            return

        self.model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=self.device)
        )
        self.model.eval()

        test_loader = get_test_loader()
        results = []

        print(f"Predicting on Test Set ({len(test_loader.dataset)} events)...")

        with torch.no_grad():
            for X, priors, _, event_ids in test_loader:
                X = X.to(self.device, non_blocking=True)
                priors = priors.to(self.device, non_blocking=True)

                # Forward Pass (Cartesian Output)
                preds_cart = self.model(X, priors)

                # Convert to Spherical
                px = preds_cart[:, 0].cpu().numpy()
                py = preds_cart[:, 1].cpu().numpy()
                pz = preds_cart[:, 2].cpu().numpy()

                az, zen = cartesian_to_spherical(px, py, pz)

                batch_res = pd.DataFrame(
                    {"event_id": event_ids, "azimuth": az, "zenith": zen}
                )
                results.append(batch_res)

        if results:
            submission_df = pd.concat(results, ignore_index=True)

            # Ensure correct column order
            submission_df = submission_df[["event_id", "azimuth", "zenith"]]

            # Save
            os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
