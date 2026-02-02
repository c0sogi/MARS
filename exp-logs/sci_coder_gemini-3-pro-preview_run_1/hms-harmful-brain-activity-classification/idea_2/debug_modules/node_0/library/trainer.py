import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.data_loader import get_dataloaders
from library.model import EEGNet1D


class Trainer:
    """
    Trainer class to manage training, validation, and inference for the EEG classification model.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Ensure output directories exist
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)

        # Initialize Model
        self.model = EEGNet1D(config=self.config).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Loss Function
        # KLDivLoss expects log-probabilities as input and probabilities as target
        # reduction='batchmean' mathematically aligns with KL Divergence definition
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def train_one_epoch(self, train_loader, scheduler):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            probs = self.model(inputs)

            # Numerical stability: Add epsilon before log
            # KLDivLoss requires log-probabilities
            log_probs = torch.log(probs + 1e-7)

            loss = self.criterion(log_probs, targets)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation loop.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                probs = self.model(inputs)
                log_probs = torch.log(probs + 1e-7)

                loss = self.criterion(log_probs, targets)

                running_loss += loss.item() * inputs.size(0)
                dataset_size += inputs.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                probs = self.model(inputs)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def fit(self, debug=False):
        """
        Main training loop with Early Stopping and Submission generation.
        """
        print(f"Initializing training on device: {self.device}")

        # Load Data
        train_loader, val_loader, test_loader = get_dataloaders(
            train_batch_size=self.config.BATCH_SIZE,
            val_batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            debug=debug,
        )

        # Scheduler Setup
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.MAX_LR,
            epochs=self.config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.config.PCT_START,
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, scheduler)

            # Validate
            val_loss = self.validate(val_loader)

            duration = time.time() - start_time

            # Print metrics (Full precision for val_loss as requested)
            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss}"
            )

            # Early Stopping & Checkpointing
            if val_loss < (best_loss - self.config.MIN_DELTA):
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                print(
                    f"Validation loss improved. Model saved to {self.config.MODEL_PATH}"
                )
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{self.config.PATIENCE}"
                )
                if patience_counter >= self.config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Loss: {best_loss}")

        # --- Submission Generation ---
        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(self.config.MODEL_PATH, map_location=self.device)
        )

        print("Generating predictions on test set...")
        predictions = self.predict(test_loader)

        # Retrieve eeg_ids from the dataset used by the loader
        # This ensures alignment even if debug mode subsampled the data
        submission_ids = test_loader.dataset.df["eeg_id"].values

        submission_df = pd.DataFrame(predictions, columns=self.config.OUTPUT_COLS)
        submission_df.insert(0, "eeg_id", submission_ids)

        submission_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")


def run_training(debug=False):
    """
    Helper function to instantiate Trainer and run fit.
    """
    trainer = Trainer()
    trainer.fit(debug=debug)
