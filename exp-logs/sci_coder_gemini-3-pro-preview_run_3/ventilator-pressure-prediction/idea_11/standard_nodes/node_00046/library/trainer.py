import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import pandas as pd
from library.config import Config
from library.model import PMNCNet


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy and torch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def masked_mae_loss(y_pred, y_true, u_out):
    """
    Computes the Mean Absolute Error (MAE) strictly on the inspiratory phase.

    Args:
        y_pred: Predicted pressure (Batch, Seq_Len)
        y_true: Actual pressure (Batch, Seq_Len)
        u_out: Expiratory valve control (Batch, Seq_Len) - 0 for inspiratory, 1 for expiratory

    Returns:
        Scalar tensor representing the masked MAE.
    """
    # Create mask: 1 where u_out == 0 (Inspiratory), 0 otherwise
    mask = (u_out == 0).float()

    # Compute absolute error
    absolute_error = torch.abs(y_pred - y_true)

    # Apply mask
    masked_error = absolute_error * mask

    # Compute mean over the number of valid inspiratory time steps
    # Add a small epsilon to denominator to avoid division by zero
    loss = masked_error.sum() / (mask.sum() + 1e-8)

    return loss


class Trainer:
    """
    Handles training, validation, and prediction for the PM-NC-Net model.
    """

    def __init__(self, train_loader, val_loader, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else Config.DEVICE

        # Initialize Model
        self.model = PMNCNet().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        # Identify index of u_out for loss calculation
        try:
            self.u_out_idx = Config.FEATURE_COLS.index("u_out")
        except ValueError:
            raise ValueError("'u_out' not found in Config.FEATURE_COLS")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Extract u_out for masking (Batch, Seq_Len)
            # u_out is a feature in batch_x at index self.u_out_idx
            u_out = batch_x[:, :, self.u_out_idx]

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(batch_x)

            # Compute Loss
            loss = masked_mae_loss(preds, batch_y, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                u_out = batch_x[:, :, self.u_out_idx]

                preds = self.model(batch_x)
                loss = masked_mae_loss(preds, batch_y, u_out)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(self):
        """
        Main training loop.
        """
        set_seed(Config.SEED)
        best_val_loss = float("inf")

        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs...")

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Update Scheduler
            self.scheduler.step(val_loss)

            # Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                saved_msg = " [Saved Best]"
            else:
                saved_msg = ""

            # Save Last Model
            torch.save(self.model.state_dict(), Config.LAST_MODEL_PATH)

            duration = time.time() - start_time

            # Print metrics in full precision
            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Train MAE: {train_loss} | "
                f"Val MAE: {val_loss} | "
                f"Time: {duration:.2f}s{saved_msg}"
            )

        print(f"Training complete. Best Validation MAE: {best_val_loss}")

    def predict(self, test_loader, test_ids):
        """
        Generates predictions for the test set using the best model.

        Args:
            test_loader: DataLoader for test data
            test_ids: Numpy array of IDs corresponding to the test data (N, 80)

        Returns:
            DataFrame with 'id' and 'pressure' columns.
        """
        # Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

        print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch_x in test_loader:
                batch_x = batch_x.to(self.device)

                # Forward pass
                preds = self.model(batch_x)

                # Move to CPU and numpy
                preds_np = preds.cpu().numpy()
                all_preds.append(preds_np)

        # Concatenate all batches: (N_breaths, 80)
        all_preds = np.concatenate(all_preds, axis=0)

        # Flatten predictions and IDs to match submission format (Row-wise)
        # test_ids is (N_breaths, 80)
        flat_preds = all_preds.flatten()
        flat_ids = test_ids.flatten()

        # Create DataFrame
        submission_df = pd.DataFrame(
            {Config.ID_COL: flat_ids, Config.TARGET_COL: flat_preds}
        )

        return submission_df


def run_training(train_loader, val_loader):
    """
    Helper function to instantiate Trainer and run fit.
    """
    trainer = Trainer(train_loader, val_loader)
    trainer.fit()
    return trainer


def generate_submission(trainer, test_loader, test_ids):
    """
    Helper function to generate and save submission.
    """
    submission_df = trainer.predict(test_loader, test_ids)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
