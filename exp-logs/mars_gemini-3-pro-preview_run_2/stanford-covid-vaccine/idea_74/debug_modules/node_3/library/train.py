import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config, set_seed
from library.data import get_loaders, process_data
from library.model import HCHSGFN
from library.loss import MCRMSELoss


class Trainer:
    def __init__(self, model, device, train_loader, val_loader):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = MCRMSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        self.best_val_score = float("inf")
        self.patience_counter = 0
        self.model_save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (inputs, pairs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            pairs = pairs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # --- Iterative Refinement Loop ---

            # Pass 1: Static features only (y_prev is implicitly zeros inside model if None)
            pred1, _ = self.model(inputs, pairs, y_prev=None)

            # Detach gradients for feedback input
            pred1_detached = pred1.detach()

            # Pass 2: With feedback
            pred2, _ = self.model(inputs, pairs, y_prev=pred1_detached)

            # Loss Calculation: Weighted sum of both passes
            # We strictly mask loss inside MCRMSELoss (only scored cols, only scored positions)
            loss = self.criterion(pred2, targets) + 0.5 * self.criterion(pred1, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()

        # Accumulators for Global MCRMSE calculation
        # We track SSE per scored column to avoid batch averaging bias
        num_scored_cols = len(Config.SCORED_TARGET_INDICES)
        column_sse = torch.zeros(num_scored_cols, device=self.device)
        total_valid_elements = 0

        with torch.no_grad():
            for inputs, pairs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                pairs = pairs.to(self.device)
                targets = targets.to(self.device)

                # Inference: Two passes
                pred1, _ = self.model(inputs, pairs, y_prev=None)
                pred2, _ = self.model(
                    inputs, pairs, y_prev=pred1
                )  # No detach needed in eval

                # Extract scored portions for metric calculation
                # Shape: (Batch, Scored_Len, Scored_Cols)
                pred_scored = pred2[
                    :, : Config.SCORED_LEN, Config.SCORED_TARGET_INDICES
                ]
                target_scored = targets[
                    :, : Config.SCORED_LEN, Config.SCORED_TARGET_INDICES
                ]

                # Compute SSE for this batch
                diff = pred_scored - target_scored
                # Sum over batch (dim 0) and sequence (dim 1)
                batch_sse = torch.sum(diff**2, dim=(0, 1))

                column_sse += batch_sse
                # Count elements per column: Batch_Size * Scored_Len
                total_valid_elements += inputs.size(0) * Config.SCORED_LEN

        # Compute Global RMSE per column
        column_mse = column_sse / total_valid_elements
        column_rmse = torch.sqrt(column_mse)

        # MCRMSE is the mean of the column RMSEs
        global_mcrmse = torch.mean(column_rmse).item()

        return global_mcrmse

    def fit(self, epochs=Config.EPOCHS):
        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_score = self.validate()

            # Print full precision as requested
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
            )

            self.scheduler.step(val_score)

            # Early Stopping and Model Checkpointing
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        print(f"Training complete. Best Val MCRMSE: {self.best_val_score}")
        return self.best_val_score


def generate_submission(model_path, test_loader, test_ids, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = HCHSGFN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Generating predictions for test set...")
    all_preds = []

    with torch.no_grad():
        for inputs, pairs in test_loader:
            inputs = inputs.to(device)
            pairs = pairs.to(device)

            # Inference: Two passes
            pred1, _ = model(inputs, pairs, y_prev=None)
            pred2, _ = model(inputs, pairs, y_prev=pred1)

            # Move to CPU
            all_preds.append(pred2.cpu().numpy())

    # Concatenate all batches: (N_samples, 5, Seq_Len)
    # Note: Model output is (B, 5, L), but submission format usually expects flattened rows per seqpos
    # Let's verify model output shape. library.model InteractionModule returns (B, 5, L).
    # We need to transpose to (B, L, 5) for easier row iteration.
    all_preds = np.concatenate(all_preds, axis=0).transpose(0, 2, 1)  # (N, 107, 5)

    # Prepare CSV data
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            # Row ID format: id_seqpos
            row_id = f"{sample_id}_{seqpos}"

            # Get values for this position
            values = sample_preds[seqpos]

            row_data = [row_id] + values.tolist()
            submission_rows.append(row_data)

    # Create DataFrame
    columns = ["id_seqpos"] + target_cols
    submission_df = pd.DataFrame(submission_rows, columns=columns)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(epochs=Config.EPOCHS, load_cached_data=True):
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get Data Loaders
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = HCHSGFN().to(device)

    # 3. Train
    trainer = Trainer(model, device, train_loader, val_loader)
    trainer.fit(epochs=epochs)

    # 4. Generate Submission
    # Retrieve test IDs from the processed data cache to align with test_loader
    # We re-load the cache dictionary to get the IDs directly
    data_cache = process_data(load_cached_data=True)
    test_ids = data_cache["test"]["ids"]

    submission_path = "./submission/submission.csv"
    generate_submission(trainer.model_save_path, test_loader, test_ids, submission_path)
