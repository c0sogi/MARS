import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, metric_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MCRMSELoss


class Trainer:
    """
    Trainer class to manage the training, validation, and prediction lifecycle
    of the RNA Degradation Prediction model.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        # Initialize Model
        self.model = RNAModel(config).to(self.device)

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

        # Loss Function
        # Optimizes MCRMSE on all 5 targets, sliced to seq_scored (68)
        self.criterion = MCRMSELoss(seq_scored=config.pred_len)

        # Best metric tracking
        self.best_score = float("inf")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (features, pair_indices, targets) in enumerate(train_loader):
            features = features.to(self.device)
            pair_indices = pair_indices.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features, pair_indices)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Runs validation and computes the official MCRMSE metric.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, pair_indices, targets in val_loader:
                features = features.to(self.device)
                pair_indices = pair_indices.to(self.device)

                outputs = self.model(features, pair_indices)

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute Metric on the 3 scored columns
        score = metric_mcrmse(all_targets, all_preds, seq_scored=self.config.pred_len)
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
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_score = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_score} | "
                f"Time: {elapsed:.2f}s"
            )

            # Early Stopping and Model Checkpointing
            if val_score < self.best_score:
                self.best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.model_save_path)
                print(f"  New best model saved! Score: {self.best_score}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        # Load best model
        print(f"Loading best model from {self.config.model_save_path}...")
        self.model.load_state_dict(
            torch.load(self.config.model_save_path, map_location=self.device)
        )
        self.model.eval()

        all_preds = []

        with torch.no_grad():
            for features, pair_indices, _ in test_loader:
                features = features.to(self.device)
                pair_indices = pair_indices.to(self.device)

                outputs = self.model(features, pair_indices)
                all_preds.append(outputs.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def generate_submission(self, predictions):
        """
        Formats predictions and saves the submission CSV.
        """
        print("Generating submission file...")

        # Load test metadata to get IDs
        test_df = pd.read_parquet(self.config.test_file)
        ids = test_df["id"].values

        # predictions shape: (N_samples, 107, 5)
        # We need to flatten this to (N_samples * 107, 6)

        data = []
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(ids):
            sample_preds = predictions[i]  # (107, 5)
            for seq_pos in range(self.config.seq_len):
                row_id = f"{sample_id}_{seq_pos}"
                row_values = sample_preds[seq_pos].tolist()
                data.append([row_id] + row_values)

        submission_df = pd.DataFrame(data, columns=["id_seqpos"] + target_cols)

        submission_df.to_csv(self.config.submission_path, index=False)
        print(f"Submission saved to {self.config.submission_path}")


def run_training(debug=False):
    """
    Main execution function.
    """
    # 1. Configuration and Seeding
    config = Config(debug=debug)
    seed_everything(config.seed)

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=debug
    )

    # 3. Trainer Initialization
    trainer = Trainer(config)

    # 4. Training
    trainer.fit(train_loader, val_loader)

    # 5. Prediction
    predictions = trainer.predict(test_loader)

    # 6. Submission
    trainer.generate_submission(predictions)


if __name__ == "__main__":
    # By default, run full training.
    # Set debug=True for quick testing during development.
    run_training(debug=False)
