import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.data import get_loaders
from library.model import RAGLUNet
from library.utils import set_seed, compute_metrics


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle of the RA-GLU-Net model.
    """

    def __init__(self, model, scaler, device):
        self.model = model.to(device)
        self.scaler = scaler
        self.device = device
        self.criterion = nn.MSELoss()

        # Initialize optimizer with weight decay as per config
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Move batch to device
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)

            # Compute loss (MSE on standardized targets)
            loss = self.criterion(outputs, batch.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(train_loader.dataset)

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average MSE loss and RMSLE metric (on original scale).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)

                # Loss on scaled data
                loss = self.criterion(outputs, batch.y)
                total_loss += loss.item() * batch.num_graphs

                # Collect for metric calculation (inverse transform later)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)

        # Concatenate and inverse transform
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        preds_original = self.scaler.inverse_transform(all_preds)
        targets_original = self.scaler.inverse_transform(all_targets)

        # Compute RMSLE
        rmsle = compute_metrics(preds_original, targets_original)

        return avg_loss, rmsle

    def fit(
        self,
        train_loader,
        val_loader,
        max_epochs=Config.MAX_EPOCHS,
        patience=Config.PATIENCE,
    ):
        """
        Main training loop with Early Stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")
        print(
            f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}"
        )

        for epoch in range(1, max_epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_rmsle = self.evaluate(val_loader)

            # Scheduler step
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            duration = time.time() - start_time

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val RMSLE: {val_rmsle:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {duration:.2f}s"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
                # print(f"  New best model saved to {Config.MODEL_CHECKPOINT_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        print(f"Loading best model from {Config.MODEL_CHECKPOINT_PATH}...")
        self.model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=self.device)
        )
        self.model.eval()

        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                all_preds.append(outputs.cpu().numpy())

        # Concatenate and inverse transform
        all_preds = np.concatenate(all_preds, axis=0)
        preds_original = self.scaler.inverse_transform(all_preds)

        return preds_original


def run_training(load_cached_data=True):
    """
    Orchestrates the entire training and submission pipeline.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # get_loaders handles processing, caching, and scaling
    train_loader, val_loader, test_loader, scaler = get_loaders(
        load_cached_data=load_cached_data, batch_size=Config.BATCH_SIZE
    )

    # 3. Model Initialization
    model = RAGLUNet(config=Config)

    # 4. Training
    trainer = Trainer(model, scaler, device)
    trainer.fit(train_loader, val_loader)

    # 5. Prediction
    predictions = trainer.predict(test_loader)

    # 6. Submission Generation
    # Load test metadata to get IDs
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Ensure predictions match test set size
    if len(predictions) != len(test_meta):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match test set size ({len(test_meta)})"
        )

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": test_meta["id"],
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Display first few rows
    print("\nSubmission Head:")
    print(submission_df.head())
