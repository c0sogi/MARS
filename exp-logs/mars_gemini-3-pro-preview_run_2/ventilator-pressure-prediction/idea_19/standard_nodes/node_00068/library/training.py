import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, WeightedL1Loss, compute_metric
from library.data_processing import prepare_data
from library.model import CWCDP_BiLSTM


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the CWCDP-BiLSTM model.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function with inspiratory/expiratory weighting
        self.criterion = WeightedL1Loss()

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing stretched over full epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for X, u_out, y in self.train_loader:
            X, u_out, y = X.to(self.device), u_out.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(X, u_out)

            # Calculate loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping to prevent exploding gradients in deep LSTMs
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate_epoch(self):
        """
        Runs validation and calculates metrics.
        Returns average weighted loss and exact inspiratory MAE.
        """
        self.model.eval()
        total_loss = 0.0
        total_insp_mae_sum = 0.0
        total_insp_count = 0
        num_batches = 0

        with torch.no_grad():
            for X, u_out, y in self.val_loader:
                X, u_out, y = (
                    X.to(self.device),
                    u_out.to(self.device),
                    y.to(self.device),
                )

                preds = self.model(X, u_out)

                # Loss calculation
                loss = self.criterion(preds, y, u_out)
                total_loss += loss.item()
                num_batches += 1

                # Metric Calculation (MAE on Inspiratory Phase)
                # compute_metric returns the mean MAE for the batch.
                # To get exact epoch MAE, we recover the sum and count.
                batch_mae = compute_metric(preds, y, u_out)

                # Count inspiratory steps in this batch (u_out == 0)
                # u_out shape is (Batch, Seq) or (Batch, Seq, 1)
                insp_mask = u_out == 0
                insp_count = insp_mask.sum().item()

                if insp_count > 0:
                    total_insp_mae_sum += batch_mae * insp_count
                    total_insp_count += insp_count

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_mae = total_insp_mae_sum / total_insp_count if total_insp_count > 0 else 0.0

        return avg_loss, avg_mae

    def fit(self, epochs, patience=25):
        """
        Main training loop with Early Stopping and Model Checkpointing.
        """
        best_mae = float("inf")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_mae = self.validate_epoch()

            # Step scheduler
            self.scheduler.step()

            # Print full precision metrics
            print(
                f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val MAE = {val_mae}"
            )

            # Checkpointing
            if val_mae < best_mae:
                best_mae = val_mae
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best Val MAE: {best_mae}"
                )
                break

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for X, u_out in test_loader:
                X, u_out = X.to(self.device), u_out.to(self.device)

                preds = self.model(X, u_out)

                # Move to CPU and flatten
                preds_np = preds.cpu().numpy().flatten()
                all_preds.append(preds_np)

        return np.concatenate(all_preds)


def run_training(epochs=Config.EPOCHS, debug=False):
    """
    Orchestrates the data preparation, training, and submission generation.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data Preparation
    # prepare_data handles caching internally
    train_loader, val_loader, test_loader, test_ids = prepare_data(
        batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=debug
    )

    # 3. Model Initialization
    model = CWCDP_BiLSTM().to(device)

    # 4. Training
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit(epochs=epochs)

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    print("Generating predictions on test set...")
    predictions = trainer.predict(test_loader)

    # 6. Submission
    # Ensure lengths match
    if len(predictions) != len(test_ids):
        print(
            f"Warning: Prediction length ({len(predictions)}) does not match ID length ({len(test_ids)})."
        )

    submission = pd.DataFrame({"id": test_ids, "pressure": predictions})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
