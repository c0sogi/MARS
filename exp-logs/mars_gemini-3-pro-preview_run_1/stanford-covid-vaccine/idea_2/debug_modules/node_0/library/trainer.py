import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from library.config import Config
from library.dataset import RNAGraphDataset
from library.model import RNAGNN


def masked_mcrmse_loss(preds, targets, mask):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    considering only the masked positions.

    Args:
        preds: (N_nodes, 5) Predicted values
        targets: (N_nodes, 5) Ground truth values
        mask: (N_nodes,) Boolean mask indicating scored positions

    Returns:
        loss: Scalar tensor representing the MCRMSE
    """
    # Filter predictions and targets based on the mask
    preds_masked = preds[mask]
    targets_masked = targets[mask]

    # Calculate MSE for each of the 5 targets (columns)
    # Adding epsilon to avoid nan gradients if mse is 0
    mse_per_col = torch.mean((preds_masked - targets_masked) ** 2, dim=0)

    # Calculate RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col + 1e-8)

    # Calculate mean of RMSEs across columns
    loss = torch.mean(rmse_per_col)

    return loss


class Trainer:
    def __init__(self, load_cached_data=True):
        """
        Initializes the Trainer with model, optimizer, scheduler, and datasets.
        """
        # Set seeds for reproducibility
        torch.manual_seed(Config.SEED)
        np.random.seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(Config.SEED)

        self.device = torch.device(Config.DEVICE)
        print(f"Using device: {self.device}")

        # Initialize Model
        self.model = RNAGNN().to(self.device)

        # Initialize Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Load Datasets
        print("Initializing Training Dataset...")
        self.train_dataset = RNAGraphDataset(
            split="train", load_cached_data=load_cached_data
        )

        print("Initializing Validation Dataset...")
        self.val_dataset = RNAGraphDataset(
            split="val", load_cached_data=load_cached_data
        )

        # Initialize DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(batch)

            # Calculate Loss
            loss = masked_mcrmse_loss(preds, batch.y, batch.mask)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self):
        """
        Runs validation on the validation set.
        Calculates the global MCRMSE over the entire dataset.
        """
        self.model.eval()

        # Accumulators for global MSE calculation
        total_squared_error = torch.zeros(Config.NUM_TARGETS, device=self.device)
        total_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                preds = self.model(batch)

                # Apply mask
                mask = batch.mask
                preds_masked = preds[mask]
                targets_masked = batch.y[mask]

                # Accumulate squared errors per column
                # Sum over the batch dimension (nodes)
                squared_diff = (preds_masked - targets_masked) ** 2
                total_squared_error += torch.sum(squared_diff, dim=0)

                # Accumulate count of valid nodes
                total_count += mask.sum().item()

        # Calculate global metrics
        if total_count == 0:
            return 0.0

        mse_per_col = total_squared_error / total_count
        rmse_per_col = torch.sqrt(mse_per_col)
        mcrmse = torch.mean(rmse_per_col).item()

        return mcrmse

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {Config.EPOCHS} epochs...")

        best_val_score = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch()
            val_score = self.validate()

            # Step the scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:02d} | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.10f} | Val MCRMSE: {val_score:.15f}"
            )

            # Checkpointing and Early Stopping
            if val_score < best_val_score:
                best_val_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"  -> New best model saved! Score: {best_val_score:.15f}")
            else:
                patience_counter += 1
                print(
                    f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {best_val_score:.15f}")

    def predict(self):
        """
        Generates predictions for the test set using the best model.
        Saves the submission file.
        """
        print("Loading best model for inference...")
        if not os.path.exists(Config.BEST_MODEL_PATH):
            raise FileNotFoundError("Best model not found. Run fit() first.")

        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        print("Initializing Test Dataset...")
        test_dataset = RNAGraphDataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        ids_list = []
        preds_list = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)

                # Forward pass
                # Output shape: (Total_Nodes_In_Batch, 5)
                out = self.model(batch)

                # Unbatching logic
                # We know every graph has exactly Config.SEQ_LEN nodes
                batch_size = len(batch.id)
                seq_len = Config.SEQ_LEN

                # Reshape to (Batch_Size, Seq_Len, 5)
                out_reshaped = out.view(batch_size, seq_len, Config.NUM_TARGETS)

                # Move to CPU
                out_np = out_reshaped.cpu().numpy()

                ids_list.extend(batch.id)
                preds_list.append(out_np)

        # Concatenate all predictions: (Total_Samples, Seq_Len, 5)
        all_preds = np.concatenate(preds_list, axis=0)

        # Format for submission
        submission_data = []
        target_cols = Config.TARGET_COLS

        print("Formatting submission...")
        for i, sample_id in enumerate(ids_list):
            sample_pred = all_preds[i]  # Shape (107, 5)

            for seq_pos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seq_pos}"
                row_values = sample_pred[seq_pos]

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(target_cols):
                    row_dict[col_name] = float(row_values[col_idx])

                submission_data.append(row_dict)

        df_sub = pd.DataFrame(submission_data)

        # Ensure output directory exists
        os.makedirs("./submission", exist_ok=True)
        output_path = "./submission/submission.csv"

        df_sub.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        print(f"Submission shape: {df_sub.shape}")
