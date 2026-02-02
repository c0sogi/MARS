import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, ensure_dir
from library.model import VentilatorNet
from library.dataset import get_data_loaders


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MaskedL1Loss(nn.Module):
    """
    Computes Mean Absolute Error (MAE) on the inspiratory phase (u_out == 0).
    This is used for the validation metric, distinct from the composite training loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, u_out):
        # Mask: 1.0 for Inspiratory (u_out=0), 0.0 for Expiratory
        mask = 1.0 - u_out

        loss = torch.abs(pred - target) * mask

        # Avoid division by zero
        mask_sum = mask.sum()
        if mask_sum < 1e-6:
            return torch.tensor(0.0, device=pred.device)

        return loss.sum() / mask_sum


def train_one_epoch(model, loader, optimizer, scheduler, device, config, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        u_out = batch["u_out"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        # The model computes the composite loss internally (Main + Aux)
        output = model(x, u_out=u_out, target=y)
        loss = output["loss"]

        # Backward pass
        loss.backward()

        # Gradient Clipping
        if config.GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), x.size(0))

    return losses.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set using MAE on inspiratory phase.
    """
    model.eval()
    metric_fn = MaskedL1Loss()
    metrics = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            u_out = batch["u_out"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            # Forward pass in inference mode
            output = model(x)
            pred = output["prediction"]

            # Compute pure MAE metric (ignoring auxiliary heads)
            mae = metric_fn(pred, y, u_out)
            metrics.update(mae.item(), x.size(0))

    return metrics.avg


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns a flattened numpy array of predictions.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)

            output = model(x)
            # Flatten the (Batch, Seq) output to 1D array
            batch_preds = output["prediction"].cpu().numpy().flatten()
            preds.append(batch_preds)

    return np.concatenate(preds)


class Trainer:
    """
    Main class to handle model training, evaluation, and submission.
    """

    def __init__(self, config: Config, debug: bool = False):
        self.config = config
        self.debug = debug
        self.device = torch.device(config.DEVICE)

        # Ensure reproducibility
        seed_everything(config.SEED)

        # Initialize DataLoaders
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            config, debug
        )

        # Initialize Model
        self.model = VentilatorNet(config).to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.LR_MAX, weight_decay=config.WEIGHT_DECAY
        )

        # Initialize Scheduler (OneCycleLR)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LR_MAX,
            steps_per_epoch=len(self.train_loader),
            epochs=config.EPOCHS,
            pct_start=config.PCT_START,
            div_factor=config.DIV_FACTOR,
            final_div_factor=config.FINAL_DIV_FACTOR,
        )

    def fit(self):
        """
        Runs the full training loop with early stopping.
        """
        best_mae = float("inf")
        patience = 7  # Early stopping patience
        counter = 0

        print(f"Starting training on {self.device} for {self.config.EPOCHS} epochs...")

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.scheduler,
                self.device,
                self.config,
                epoch,
            )

            # Validate
            val_mae = evaluate(self.model, self.val_loader, self.device)

            elapsed = time.time() - start_time

            # Print metrics (Full precision)
            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"Time: {elapsed:.1f}s | "
                f"Train Loss: {train_loss:.10f} | "
                f"Val MAE: {val_mae:.10f}"
            )

            # Checkpoint & Early Stopping
            if val_mae < best_mae:
                best_mae = val_mae
                counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                print(f"Saved best model with MAE: {best_mae:.10f}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    def generate_submission(self):
        """
        Loads the best model, generates predictions on the test set,
        and saves the submission file.
        """
        if not os.path.exists(self.config.MODEL_PATH):
            print("No model checkpoint found. Skipping submission.")
            return

        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(self.config.MODEL_PATH, map_location=self.device)
        )

        print("Generating predictions on test set...")
        predictions = predict(self.model, self.test_loader, self.device)

        # Load sample submission to get IDs and structure
        sub_df = pd.read_csv(self.config.SAMPLE_SUBMISSION)

        # Verify length
        if len(predictions) != len(sub_df):
            print(
                f"Warning: Prediction length {len(predictions)} does not match submission length {len(sub_df)}"
            )

        # Assign predictions
        sub_df["pressure"] = predictions

        # Save to Config path
        sub_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")

        # Save to specific requested path
        alt_path = "./submission/submission.csv"
        ensure_dir(alt_path)
        sub_df.to_csv(alt_path, index=False)
        print(f"Submission also saved to {alt_path}")
