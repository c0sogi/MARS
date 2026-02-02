import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from library.config import Config
from library.utils import mcrmse, get_device


def masked_mse_loss(preds, targets, scored_len=Config.SEQ_SCORED):
    """
    Computes MSE loss only on the scored positions of the sequence.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Channels)
        targets (torch.Tensor): Ground truth of shape (Batch, Scored_Len, Channels)
        scored_len (int): The number of positions to score from the start.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Slice predictions to match the length of targets
    # preds: (B, 107, 5) -> (B, 68, 5)
    preds_sliced = preds[:, :scored_len, :]

    # Compute MSE
    loss = nn.MSELoss()(preds_sliced, targets)
    return loss


class Engine:
    """
    Engine class to handle training, validation, and inference.
    """

    def __init__(self, model, optimizer=None, scheduler=None, device=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else get_device()
        self.model.to(self.device)

        # Identify indices of columns that are actually scored for the metric
        # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

    def train_one_epoch(self, dataloader):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            # Move inputs to device
            seq = batch["sequence"].to(self.device)
            struct = batch["structure"].to(self.device)
            loop = batch["loop_type"].to(self.device)
            targets = batch["targets"].to(self.device)  # (B, 68, 5)

            # Forward pass
            # Output: (B, 107, 5)
            preds = self.model(seq, struct, loop)

            # Calculate loss on the valid portion (first 68 bases)
            loss = masked_mse_loss(preds, targets, Config.SEQ_SCORED)

            # Backward pass
            if self.optimizer:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        return avg_loss

    def validate(self, dataloader):
        """
        Evaluates the model on the validation set using MCRMSE.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                seq = batch["sequence"].to(self.device)
                struct = batch["structure"].to(self.device)
                loop = batch["loop_type"].to(self.device)
                targets = batch["targets"].cpu().numpy()  # (B, 68, 5)

                preds = self.model(seq, struct, loop)
                preds = preds.cpu().numpy()  # (B, 107, 5)

                # Slice predictions to match targets for scoring
                preds_sliced = preds[:, : Config.SEQ_SCORED, :]

                all_preds.append(preds_sliced)
                all_targets.append(targets)

        # Concatenate all batches
        y_pred = np.concatenate(all_preds, axis=0)  # (N, 68, 5)
        y_true = np.concatenate(all_targets, axis=0)  # (N, 68, 5)

        # Filter only the scored columns for the metric
        # y_pred shape: (N, 68, 5) -> select specific channels
        y_pred_scored = y_pred[:, :, self.scored_indices]
        y_true_scored = y_true[:, :, self.scored_indices]

        # Flatten for MCRMSE calculation if necessary, or pass as is.
        # The utils.mcrmse expects (N, C) usually, but here we have (N, L, C).
        # We need to reshape to (N*L, C) to compute column-wise RMSE over all predictions.
        N, L, C = y_pred_scored.shape
        y_pred_flat = y_pred_scored.reshape(-1, C)
        y_true_flat = y_true_scored.reshape(-1, C)

        score = mcrmse(y_true_flat, y_pred_flat)
        return score

    def fit(
        self, train_loader, val_loader, epochs=Config.EPOCHS, patience=Config.PATIENCE
    ):
        """
        Runs the full training process with Early Stopping.
        """
        best_score = float("inf")
        patience_counter = 0
        best_model_state = None

        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_score = self.validate(val_loader)

            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_score)
                else:
                    self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.20f}"
            )

            # Early Stopping Check
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                best_model_state = self.model.state_dict()
                # Save best model temporarily
                torch.save(
                    best_model_state, os.path.join(Config.WORKING_DIR, "best_model.pth")
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with Val MCRMSE: {best_score:.20f}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set and creates the submission file.
        """
        self.model.eval()
        all_preds = []
        all_ids = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                seq = batch["sequence"].to(self.device)
                struct = batch["structure"].to(self.device)
                loop = batch["loop_type"].to(self.device)
                ids = batch["id"]

                # Predict
                preds = self.model(seq, struct, loop)  # (B, 107, 5)
                preds = preds.cpu().numpy()

                all_preds.append(preds)
                all_ids.extend(ids)

        # Concatenate
        # Shape: (N_samples, 107, 5)
        predictions = np.concatenate(all_preds, axis=0)

        # Prepare submission data
        # We need to flatten: one row per sequence position
        # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

        submission_rows = []
        target_cols = Config.TARGET_COLS

        for i, sample_id in enumerate(all_ids):
            sample_preds = predictions[i]  # (107, 5)

            for seqpos in range(Config.SEQ_LEN):
                # Row ID
                row_id = f"{sample_id}_{seqpos}"

                # Values
                vals = sample_preds[seqpos]

                # Create dict
                row_data = {"id_seqpos": row_id}
                for j, col in enumerate(target_cols):
                    row_data[col] = float(vals[j])

                submission_rows.append(row_data)

        # Create DataFrame
        df_sub = pd.DataFrame(submission_rows)

        # Ensure column order
        cols = ["id_seqpos"] + target_cols
        df_sub = df_sub[cols]

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
