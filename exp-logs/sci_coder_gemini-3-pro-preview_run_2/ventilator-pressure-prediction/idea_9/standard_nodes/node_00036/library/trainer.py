import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from library.config import Config
from library.utils import seed_everything, get_device, MetricMonitor
from library.data_processing import prepare_dataloaders
from library.model import SC_GI_BiLSTM


# ==========================================
# Loss Function
# ==========================================
class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss that assigns different weights to inspiratory and expiratory phases.
    Inspiratory (u_out=0): Weight = 1.0
    Expiratory (u_out=1): Weight = 0.1
    """

    def __init__(self, w_insp=Config.W_INSPIRATORY, w_exp=Config.W_EXPIRATORY):
        super().__init__()
        self.w_insp = w_insp
        self.w_exp = w_exp
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, preds, targets, u_out):
        """
        Args:
            preds: (Batch, Seq)
            targets: (Batch, Seq)
            u_out: (Batch, Seq) - Binary feature indicating expiratory phase
        """
        loss = self.l1(preds, targets)

        # u_out is 0 for inspiratory, 1 for expiratory
        # We assume u_out is strictly 0.0 or 1.0
        weights = torch.where(u_out == 0, self.w_insp, self.w_exp)

        weighted_loss = loss * weights
        return weighted_loss.mean()


# ==========================================
# Trainer Class
# ==========================================
class Trainer:
    def __init__(self, model, device, optimizer, scheduler, criterion):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.best_mae = float("inf")

        # Identify index of u_out in features for loss calculation and validation
        try:
            self.u_out_idx = Config.SELECTED_FEATURES.index("u_out")
        except ValueError:
            raise ValueError("'u_out' not found in Config.SELECTED_FEATURES")

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        metric_monitor = MetricMonitor()

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            # Unpack batch
            x, y = batch
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)

            # Extract u_out for loss weighting (Batch, Seq)
            u_out = x[:, :, self.u_out_idx]

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(x)

            # Calculate Loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimizer Step
            self.optimizer.step()

            # Update metrics
            metric_monitor.update("Loss", loss.item())

            # Update progress bar
            pbar.set_postfix(Loss=f"{metric_monitor.get_avg('Loss'):.6f}")

        return metric_monitor.get_avg("Loss")

    def validate(self, val_loader):
        self.model.eval()
        metric_monitor = MetricMonitor()

        # We don't need gradients for validation
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)

                # Extract u_out
                u_out = x[:, :, self.u_out_idx]

                # Forward pass
                preds = self.model(x)

                # Calculate Metric: MAE on Inspiratory Phase ONLY (u_out == 0)
                mask = u_out == 0

                # Avoid division by zero if batch has no inspiratory phase (unlikely but safe)
                if mask.sum() > 0:
                    mae = torch.abs(preds[mask] - y[mask]).mean()
                    metric_monitor.update("MAE", mae.item(), n=mask.sum().item())

        return metric_monitor.get_avg("MAE")

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_mae = self.validate(val_loader)

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            # Print Metrics (Full Precision)
            print(f"Epoch {epoch} | Train Loss: {train_loss} | Val MAE: {val_mae}")

            # Checkpoint & Early Stopping
            if val_mae < self.best_mae:
                print(
                    f"Validation MAE improved from {self.best_mae} to {val_mae}. Saving model..."
                )
                self.best_mae = val_mae
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.WORKING_DIR, "best_model.pth"),
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered after {Config.PATIENCE} epochs without improvement."
                    )
                    break

        print(f"Training complete. Best Val MAE: {self.best_mae}")

    def predict(self, test_loader):
        print("Loading best model for inference...")
        checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                x = batch  # test_loader returns only x
                x = x.to(self.device, dtype=torch.float32)

                preds = self.model(x)

                # Move to CPU and flatten
                # preds shape: (Batch, Seq) -> (Batch * Seq)
                preds_flat = preds.cpu().numpy().flatten()
                predictions.extend(preds_flat)

        return np.array(predictions)


# ==========================================
# Main Execution
# ==========================================
def run_training():
    # 1. Setup
    seed_everything()
    device = get_device()
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders()

    # 3. Model
    print("Initializing Model...")
    model = SC_GI_BiLSTM().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = WeightedL1Loss()

    # 5. Training
    trainer = Trainer(model, device, optimizer, scheduler, criterion)
    trainer.fit(train_loader, val_loader)

    # 6. Inference
    print("Generating predictions...")
    preds = trainer.predict(test_loader)

    # 7. Submission Generation
    print("Creating submission file...")
    # Load test metadata to map predictions to IDs
    # Note: The model processes data sorted by [breath_id, time_step]
    # We must ensure metadata is sorted identically before assigning predictions.

    test_meta = pd.read_csv(Config.TEST_META)

    # Sort metadata to match model output order (breath_id, id/time)
    # Assuming 'id' is sequential within a breath, sorting by breath_id then id
    # aligns with the data processing logic.
    test_meta = test_meta.sort_values(["breath_id", "id"])

    # Verify lengths match
    if len(preds) != len(test_meta):
        print(
            f"Warning: Prediction length {len(preds)} does not match metadata length {len(test_meta)}."
        )
        # This might happen if data was truncated in prepare_dataloaders (e.g. debug mode)
        # We truncate metadata to match
        test_meta = test_meta.iloc[: len(preds)]

    # Assign predictions
    test_meta["pressure"] = preds

    # Sort by ID for submission format
    submission = test_meta.sort_values("id")[["id", "pressure"]]

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
