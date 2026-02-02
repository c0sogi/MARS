import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, calculate_global_mcrmse
from library.data import get_dataloaders
from library.model import DensePartnerAwareNet


class Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

        # Initialize Model
        self.model = DensePartnerAwareNet().to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Loss function component (MSE)
        self.mse_criterion = nn.MSELoss(reduction="none")

    def criterion(self, preds, targets):
        """
        MCRMSE Loss computed only on scored columns and scored sequence length.
        """
        # preds: (Batch, Seq_Len, 5)
        # targets: (Batch, Seq_Len, 5)

        # Slice to scored length (68) and scored columns
        preds_sliced = preds[:, : Config.SEQ_SCORED, self.scored_indices]
        targets_sliced = targets[:, : Config.SEQ_SCORED, self.scored_indices]

        # Compute MSE per element
        mse = self.mse_criterion(preds_sliced, targets_sliced)

        # Average over batch and sequence length to get MSE per column
        mse_per_col = torch.mean(mse, dim=(0, 1))

        # RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col + 1e-8)

        # Mean of RMSEs
        loss = torch.mean(rmse_per_col)

        return loss

    def train_one_epoch(self, loader):
        self.model.train()
        running_loss = 0.0

        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            partner_indices = batch["partner_indices"].to(self.device)
            # Targets come as (Batch, 5, Seq_Len), need (Batch, Seq_Len, 5)
            targets = batch["targets"].to(self.device).permute(0, 2, 1)

            self.optimizer.zero_grad()

            preds = self.model(inputs, partner_indices)

            loss = self.criterion(preds, targets)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        return running_loss / len(loader.dataset)

    def validate(self, loader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(self.device)
                partner_indices = batch["partner_indices"].to(self.device)
                targets = batch["targets"].to(self.device).permute(0, 2, 1)

                preds = self.model(inputs, partner_indices)

                # Move to CPU for metric calculation
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Filter for metric calculation: Only scored positions and scored columns
        # Shape: (N_samples, Seq_Len, 5) -> slice -> (N_samples, 68, 3)
        preds_filtered = all_preds[:, : Config.SEQ_SCORED, self.scored_indices]
        targets_filtered = all_targets[:, : Config.SEQ_SCORED, self.scored_indices]

        score = calculate_global_mcrmse(preds_filtered, targets_filtered)
        return score

    def run_training(self):
        set_seed(Config.SEED)

        train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

        best_score = float("inf")
        patience = 10
        patience_counter = 0
        model_path = Config.get_model_path()

        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch(train_loader)
            val_score = self.validate(val_loader)

            self.scheduler.step(val_score)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
            )

            if val_score < best_score:
                best_score = val_score
                torch.save(self.model.state_dict(), model_path)
                patience_counter = 0
                print(f"  New best model saved! Score: {best_score:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Best Val MCRMSE: {best_score:.6f}")

    def generate_submission(self):
        print("Generating submission...")
        set_seed(Config.SEED)

        # Load Data
        _, _, test_loader = get_dataloaders(load_cached_data=True)

        # Load Best Model
        model_path = Config.get_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(self.device)
                partner_indices = batch["partner_indices"].to(self.device)
                ids = batch["ids"]

                # Preds: (Batch, Seq_Len, 5)
                preds = self.model(inputs, partner_indices)
                preds = preds.cpu().numpy()

                ids_list.extend(ids)
                preds_list.append(preds)

        # Concatenate all predictions: (N_Test, 107, 5)
        all_preds = np.concatenate(preds_list, axis=0)

        # Flatten for submission
        # We need rows for every position (0 to 106) for every ID
        submission_data = []

        for i, sample_id in enumerate(ids_list):
            sample_preds = all_preds[i]  # (107, 5)

            for seq_pos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seq_pos}"
                row_preds = sample_preds[seq_pos]

                # Create dictionary for the row
                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(Config.TARGET_COLS):
                    row_dict[col_name] = float(row_preds[col_idx])

                submission_data.append(row_dict)

        # Create DataFrame
        submission_df = pd.DataFrame(submission_data)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training_pipeline():
    """
    Entry point function to run the full training and submission pipeline.
    """
    trainer = Trainer()
    trainer.run_training()
    trainer.generate_submission()
