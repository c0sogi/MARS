import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from library.config import Config
from library.utils import AverageMeter, save_checkpoint, load_checkpoint
from library.dataset import get_data_loaders
from library.model import BiLSTMRegressor


class Trainer:
    """
    Trainer class to handle training, validation, and inference for the Ventilator Pressure Prediction task.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, test_ids):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_ids = test_ids

        # Optimizer: AdamW for better regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Reduce LR when validation loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        # Loss Function: L1 Loss (Mean Absolute Error)
        self.criterion = nn.L1Loss()

        # Best score tracking (initialize to infinity)
        self.best_score = float("inf")

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        Returns the average loss for the epoch.
        """
        self.model.train()
        losses = AverageMeter()

        for x, y in self.train_loader:
            x = x.to(Config.DEVICE)
            y = y.to(Config.DEVICE)

            # Forward pass
            preds = self.model(x)

            # Calculate loss over the entire breath sequence
            loss = self.criterion(preds, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), x.size(0))

        return losses.avg

    def validate(self):
        """
        Runs validation on the validation set.
        Returns:
            avg_loss: L1 Loss over the entire breath (for scheduler).
            avg_metric: MAE over the inspiratory phase (for model selection).
        """
        self.model.eval()
        losses = AverageMeter()
        metric_score = AverageMeter()

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(Config.DEVICE)
                y = y.to(Config.DEVICE)

                preds = self.model(x)

                # Loss over entire breath
                loss = self.criterion(preds, y)
                losses.update(loss.item(), x.size(0))

                # Metric: MAE on inspiratory phase only (where u_out == 0)
                # u_out is the 3rd feature (index 2) in the input tensor
                u_out = x[:, :, 2]
                mask = u_out == 0

                # Apply mask to select only inspiratory phase time steps
                masked_preds = preds[mask]
                masked_y = y[mask]

                if masked_preds.numel() > 0:
                    mae = torch.abs(masked_preds - masked_y).mean()
                    # Update metric meter, weighted by number of valid samples in batch
                    metric_score.update(mae.item(), masked_preds.numel())

        return losses.avg, metric_score.avg

    def fit(self):
        """
        Main training loop with Early Stopping and Scheduler.
        """
        print(f"Starting training on device: {Config.DEVICE}")
        early_stop_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metric = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Metric: {val_metric} | "
                f"LR: {self.optimizer.param_groups[0]['lr']}"
            )

            # Update scheduler based on Validation Loss
            self.scheduler.step(val_loss)

            # Checkpoint and Early Stopping based on Competition Metric
            if val_metric < self.best_score:
                print(
                    f"Validation Metric Improved ({self.best_score} -> {val_metric}). Saving model..."
                )
                self.best_score = val_metric
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_score": self.best_score,
                    }
                )
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(
                    f"No improvement. Early stopping counter: {early_stop_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    def predict(self):
        """
        Generates predictions for the test set using the best model.
        Saves the result to submission.csv.
        """
        print("Loading best model for inference...")
        load_checkpoint(self.model, filename=Config.MODEL_CHECKPOINT)
        self.model.eval()

        predictions = []

        print("Generating predictions...")
        with torch.no_grad():
            for x in self.test_loader:
                x = x.to(Config.DEVICE)
                preds = self.model(x)
                # preds shape: (Batch, Seq_Len)
                predictions.append(preds.cpu().numpy())

        # Concatenate all batches: (Num_Breaths, Seq_Len)
        predictions = np.concatenate(predictions, axis=0)

        # Flatten to 1D array: (Num_Breaths * Seq_Len,)
        # This matches the order of test_ids which is also flattened by the dataset loader
        predictions = predictions.flatten()

        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_sub = pd.DataFrame({"id": self.test_ids, "pressure": predictions})

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")


def run_training(load_cached_data=True):
    """
    Orchestrates the data loading, model initialization, training, and prediction.

    Args:
        load_cached_data (bool): Whether to try loading pre-processed data from cache.
    """
    # 1. Get Data Loaders
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = BiLSTMRegressor()

    # 3. Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, test_ids)

    # 4. Fit Model
    trainer.fit()

    # 5. Generate Submission
    trainer.predict()
