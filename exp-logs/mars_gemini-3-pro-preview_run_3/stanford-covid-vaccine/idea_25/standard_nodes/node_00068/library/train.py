import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model import RNAModel
from library.data import get_dataloaders
from library.utils import seed_everything, calculate_global_mcrmse


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle of the RNA degradation model.
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Initialize Model
        self.model = RNAModel(config).to(self.device)

        # Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.T_max, eta_min=config.eta_min
        )

        # Loss Function
        # We use MSE as a stable surrogate for optimizing MCRMSE.
        # Since MCRMSE is the mean of RMSEs, and we treat columns equally,
        # minimizing MSE (Mean Squared Error) is effectively minimizing the objective.
        self.criterion = nn.MSELoss()

        # State tracking
        self.best_score = float("inf")

    def train_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Move data to device
            inputs = batch["inputs"].to(self.device)
            bpp_indices = batch["bpp_indices"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, bpp_indices)

            # Slice to scored sequence length for loss calculation
            # Ground truth is only valid for the first seq_scored positions (68)
            # The remaining positions (68-107) are padded with zeros and should not contribute to loss
            outputs_sliced = outputs[:, : self.config.seq_scored, :]
            targets_sliced = targets[:, : self.config.seq_scored, :]

            # Compute Loss
            loss = self.criterion(outputs_sliced, targets_sliced)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Mandatory for stability in deep BiGRUs)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_val
            )

            # Optimizer Step
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set using the official MCRMSE metric.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(self.device)
                bpp_indices = batch["bpp_indices"].to(self.device)
                targets = batch["targets"]  # Keep on CPU for aggregation

                outputs = self.model(inputs, bpp_indices)

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())

        # Concatenate all batches to compute global metric
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Filter for scored columns
        # We slice the predictions and targets to include only [reactivity, deg_Mg_pH10, deg_Mg_50C]
        # This aligns the metric calculation with the competition scoring rules.
        scored_indices = getattr(self.config, "scored_cols_indices", [0, 1, 3])
        all_preds = all_preds[:, :, scored_indices]
        all_targets = all_targets[:, :, scored_indices]

        # Calculate Global MCRMSE
        # The utility function handles slicing to seq_scored (68) internally
        score = calculate_global_mcrmse(all_preds, all_targets, self.config.seq_scored)

        return score

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_score = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()

            end_time = time.time()
            epoch_time = end_time - start_time

            # Print metrics (Full precision for val_score as requested)
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_score} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Checkpoint & Early Stopping
            if val_score < self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.config.model_save_path)
                print(f"New best model saved with MCRMSE: {self.best_score}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Val MCRMSE: {self.best_score}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        # Load best model
        if os.path.exists(self.config.model_save_path):
            self.model.load_state_dict(
                torch.load(self.config.model_save_path, map_location=self.device)
            )
            print("Loaded best model for inference.")
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()
        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(self.device)
                bpp_indices = batch["bpp_indices"].to(self.device)
                ids = batch["id"]

                outputs = self.model(inputs, bpp_indices)

                preds_list.append(outputs.cpu().numpy())
                ids_list.extend(ids)

        preds = np.concatenate(preds_list, axis=0)  # Shape: (N, 107, 5)
        return ids_list, preds

    def generate_submission(self, ids, preds):
        """
        Formats predictions and saves to submission.csv.
        """
        # preds shape: (N, 107, 5)
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        num_samples = len(ids)
        seq_len = self.config.seq_len

        # Create id_seqpos column
        # Expand ids: [id1, id1... (107 times), id2, ...]
        expanded_ids = np.repeat(ids, seq_len)

        # Create seqpos indices: [0, 1, ... 106, 0, 1, ... 106, ...]
        seq_indices = np.tile(np.arange(seq_len), num_samples)

        # Combine to "id_seqpos"
        id_seqpos = [f"{i}_{s}" for i, s in zip(expanded_ids, seq_indices)]

        # Flatten predictions: (N, 107, 5) -> (N*107, 5)
        flat_preds = preds.reshape(-1, 5)

        # Create DataFrame
        submission_df = pd.DataFrame(flat_preds, columns=target_cols)
        submission_df.insert(0, "id_seqpos", id_seqpos)

        # Save
        submission_df.to_csv(self.config.submission_path, index=False)
        print(f"Submission saved to {self.config.submission_path}")


def main():
    """
    Main execution function.
    """
    config = Config()
    seed_everything(config.seed)

    # Load Data (with caching enabled)
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # Initialize Trainer
    trainer = Trainer(config)

    # Train
    trainer.fit(train_loader, val_loader)

    # Predict
    ids, preds = trainer.predict(test_loader)

    # Generate Submission
    trainer.generate_submission(ids, preds)
