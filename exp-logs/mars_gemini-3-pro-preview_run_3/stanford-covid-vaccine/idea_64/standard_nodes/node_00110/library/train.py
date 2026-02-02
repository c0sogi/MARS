import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders
from library.model import HC_BD_BiGRU


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Computes MCRMSE across all 5 target columns for training.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        """
        Args:
            preds: (Batch, SeqLen_Pred, 5)
            targets: (Batch, SeqLen_Target, 5)
        """
        # Slice predictions to match target length (usually 68)
        seq_scored = targets.shape[1]
        preds_sliced = preds[:, :seq_scored, :]

        # Calculate MSE per column (averaging over batch and sequence length)
        # Flatten batch and sequence dimensions: (B*L, 5)
        mse = torch.mean((preds_sliced - targets) ** 2, dim=(0, 1))

        # Calculate RMSE per column
        rmse = torch.sqrt(mse)

        # Average RMSE across columns
        loss = torch.mean(rmse)

        return loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = config.DEVICE

        # Move model to device
        self.model.to(self.device)

        # Loss function
        self.criterion = MCRMSELoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS, eta_min=1e-6
        )

        # Tracking
        self.best_score = float("inf")
        self.early_stop_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            # Move data to device
            sequence = batch["sequence"].to(self.device)
            pair_indices = batch["pair_indices"].to(self.device)
            pair_mask = batch["pair_mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(sequence, pair_indices, pair_mask)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.MAX_GRAD_NORM
            )

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                sequence = batch["sequence"].to(self.device)
                pair_indices = batch["pair_indices"].to(self.device)
                pair_mask = batch["pair_mask"].to(self.device)
                targets = batch["targets"].cpu().numpy()

                outputs = self.model(sequence, pair_indices, pair_mask)
                preds = outputs.cpu().numpy()

                all_preds.append(preds)
                all_targets.append(targets)

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate MCRMSE using the utility function (handles slicing and column filtering)
        score = calculate_mcrmse(all_preds, all_targets)
        return score

    def fit(self):
        print(f"Starting training on {self.device}...")

        for epoch in range(config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_score = self.validate()

            # Step scheduler
            self.scheduler.step()

            end_time = time.time()
            epoch_time = end_time - start_time

            # Print metrics
            print(
                f"Epoch {epoch+1}/{config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_score} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Checkpointing and Early Stopping
            if val_score < self.best_score:
                self.best_score = val_score
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"New best model saved with MCRMSE: {self.best_score}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= config.PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Validation MCRMSE: {self.best_score}")

    def generate_submission(self):
        print("Generating submission...")

        # Load best model
        if os.path.exists(config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(config.MODEL_SAVE_PATH, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()
        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in self.test_loader:
                sequence = batch["sequence"].to(self.device)
                pair_indices = batch["pair_indices"].to(self.device)
                pair_mask = batch["pair_mask"].to(self.device)
                ids = batch["id"]

                outputs = self.model(sequence, pair_indices, pair_mask)
                preds = outputs.cpu().numpy()  # (B, 107, 5)

                ids_list.extend(ids)
                preds_list.append(preds)

        all_preds = np.concatenate(preds_list, axis=0)

        # Prepare submission data
        submission_data = []
        target_cols = config.TARGET_COLS

        for i, sample_id in enumerate(ids_list):
            sample_preds = all_preds[i]  # (107, 5)
            for seqpos in range(config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_preds[seqpos].tolist()
                # Row format: id_seqpos, val1, val2, val3, val4, val5
                submission_data.append([row_id] + row_values)

        columns = ["id_seqpos"] + target_cols
        submission_df = pd.DataFrame(submission_data, columns=columns)

        # Save
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")


def run_training(load_cached_data=True):
    """
    Main entry point for training and submission generation.
    """
    # Reproducibility
    seed_everything(config.SEED)

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Initialize Model
    model = HC_BD_BiGRU()

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader)

    # Train
    trainer.fit()

    # Generate Submission
    trainer.generate_submission()
