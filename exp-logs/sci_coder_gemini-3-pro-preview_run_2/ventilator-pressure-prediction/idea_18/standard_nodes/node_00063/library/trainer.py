import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed


class Trainer:
    """
    Trainer class for the WCMI-BiLSTM model.
    Manages training, validation, early stopping, and inference.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.best_val_mae = float("inf")
        # Ensure reproducibility
        set_seed(Config.SEED)

    def weighted_l1_loss(self, pred, target, u_out):
        """
        Calculates Weighted L1 Loss:
        - Weight 1.0 for Inspiratory phase (u_out == 0)
        - Weight 0.1 for Expiratory phase (u_out == 1)
        """
        error = torch.abs(pred - target)
        # u_out is 0 for inspiratory, 1 for expiratory
        weights = 1.0 * (1 - u_out) + 0.1 * u_out
        return (error * weights).mean()

    def train_epoch(self, train_loader, optimizer):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            X, y, u_out = batch
            X, y, u_out = X.to(self.device), y.to(self.device), u_out.to(self.device)

            optimizer.zero_grad()
            pred = self.model(X)
            loss = self.weighted_l1_loss(pred, y, u_out)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Metric: Mean Absolute Error (MAE) on Inspiratory Phase (u_out == 0).
        """
        self.model.eval()
        total_mae = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                X, y, u_out = batch
                X, y, u_out = (
                    X.to(self.device),
                    y.to(self.device),
                    u_out.to(self.device),
                )

                pred = self.model(X)

                # Mask for inspiratory phase (u_out == 0)
                mask = u_out == 0
                if mask.sum() > 0:
                    # Absolute error sum for inspiratory steps
                    mae_sum = torch.abs(pred[mask] - y[mask]).sum().item()
                    total_mae += mae_sum
                    count += mask.sum().item()

        return total_mae / count if count > 0 else 0.0

    def fit(
        self, train_loader, val_loader, epochs=Config.EPOCHS, lr=Config.LR, patience=25
    ):
        """
        Executes the training pipeline with AdamW, Cosine Annealing, and Early Stopping.
        """
        print(f"Initializing training on {self.device}...")

        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
        )

        # Stretched Horizon: T_max set to 200 (epochs) for epoch-level stepping
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        no_improve_epochs = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_mae = self.validate(val_loader)

            # Step scheduler after each epoch
            scheduler.step()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val MAE (Inspiratory): {val_mae}"
            )

            # Checkpoint and Early Stopping
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), best_model_path)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best Val MAE: {self.best_val_mae}"
                )
                break

        # Load best weights
        if os.path.exists(best_model_path):
            print(f"Loading best model weights from {best_model_path}")
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )

    def predict(self, test_loader):
        """
        Generates flattened predictions for the test set.
        """
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X in test_loader:
                X = X.to(self.device)
                pred = self.model(X)
                preds.append(pred.cpu().numpy().flatten())

        return np.concatenate(preds)

    def generate_submission(self, test_loader, test_ids):
        """
        Generates predictions and saves the submission file.
        """
        print("Generating submission...")
        predictions = self.predict(test_loader)

        if len(predictions) != len(test_ids):
            print(
                f"Warning: Prediction count {len(predictions)} does not match ID count {len(test_ids)}"
            )

        submission = pd.DataFrame({"id": test_ids, "pressure": predictions})

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
