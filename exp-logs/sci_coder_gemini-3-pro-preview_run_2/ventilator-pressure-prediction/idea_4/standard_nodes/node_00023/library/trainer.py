import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.utils import seed_everything, get_device, compute_metric
from library.dataset import get_dataloaders
from library.model import ResBiLSTM


class Trainer:
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 512,
        num_layers: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        dropout: float = 0.1,
        epochs: int = 50,
        patience: int = 15,
        batch_size: int = 128,
        seed: int = 42,
    ):
        """
        Initializes the Trainer with model, optimizer, scheduler, and criterion.
        """
        self.device = get_device()
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed

        seed_everything(seed)

        # Initialize the Deep Residual BiLSTM Model
        self.model = ResBiLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            dropout=dropout,
        ).to(self.device)

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler: Cosine Annealing
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Criterion: L1 Loss (MAE) on the full sequence
        # We will use a custom weighted loss in the loop, but keep this for validation logging
        self.criterion = nn.L1Loss()

        # State for early stopping
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """
        Executes one training epoch with Weighted L1 Loss (Inspiratory focus).
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            X = batch["X"].to(self.device)
            y = batch["y"].to(self.device)
            u_out = batch["u_out"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(X)
            y_pred = y_pred.squeeze(-1)

            # Weighted L1 Loss
            # Weight: 1.0 for Inspiratory (u_out=0), 0.1 for Expiratory (u_out=1)
            # This focuses optimization on the metric while keeping context for the LSTM
            weights = 1.0 - 0.9 * u_out
            loss_unreduced = torch.abs(y_pred - y)
            loss = (loss_unreduced * weights).mean()

            # Backward pass
            loss.backward()

            # Gradient Clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average Loss (Full Sequence) and Average Metric (Inspiratory Phase MAE).
        """
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                X = batch["X"].to(self.device)
                y = batch["y"].to(self.device)
                u_out = batch["u_out"].to(self.device)

                y_pred = self.model(X)

                # Validation Loss (Full sequence L1 for general health check)
                loss = self.criterion(y_pred.squeeze(-1), y)
                total_loss += loss.item()

                # Validation Metric (Inspiratory Phase MAE only)
                metric = compute_metric(y_pred, y, u_out)
                total_metric += metric
                num_batches += 1

        return total_loss / num_batches, total_metric / num_batches

    def fit(self, data_dir="./input", cache_dir="./working/idea_4/"):
        """
        Orchestrates the training process with early stopping.
        """
        print(f"Initializing training on device: {self.device}")

        # Load Data (handles caching internally)
        train_loader, val_loader, _ = get_dataloaders(
            data_dir=data_dir,
            batch_size=self.batch_size,
            load_cached_data=True,
            cache_dir=cache_dir,
        )

        best_val_metric = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metric = self.validate(val_loader)

            self.scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Metric (Insp MAE): {val_metric}"
            )

            # Early Stopping Check
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                self.best_model_state = self.model.state_dict()
                patience_counter = 0

                # Save checkpoint
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(
                    self.best_model_state, os.path.join(cache_dir, "best_model.pth")
                )
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            print(f"Restoring best model with Val Metric: {best_val_metric}")
            self.model.load_state_dict(self.best_model_state)

    def predict(self, data_dir="./input", cache_dir="./working/idea_4/"):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("Starting inference...")

        # Load Data (Test Loader)
        _, _, test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=self.batch_size,
            load_cached_data=True,
            cache_dir=cache_dir,
        )

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                X = batch["X"].to(self.device)

                # Forward pass
                y_pred = self.model(X)

                # y_pred shape: (Batch, Seq_Len, 1) -> Flatten to (Batch * Seq_Len)
                predictions.append(y_pred.cpu().numpy().flatten())

        all_preds = np.concatenate(predictions)

        # Prepare Submission DataFrame
        # We need to map predictions back to IDs.
        # The dataset logic sorts by breath_id and time_step.
        # We must read test.csv and apply the same sort to get the correct ID order.
        print("Mapping predictions to IDs...")
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        test_df = test_df.sort_values(["breath_id", "time_step"])

        # Handle potential truncation in dataset loader (if length % 80 != 0)
        # Though official test set is usually perfectly divisible.
        if len(all_preds) != len(test_df):
            SEQ_LEN = 80
            num_breaths = len(test_df) // SEQ_LEN
            test_df = test_df.iloc[: num_breaths * SEQ_LEN]
            print(f"Adjusted dataframe length to {len(test_df)} to match predictions.")

        submission = pd.DataFrame({"id": test_df["id"].values, "pressure": all_preds})

        # Save
        os.makedirs("./submission", exist_ok=True)
        sub_path = "./submission/submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
