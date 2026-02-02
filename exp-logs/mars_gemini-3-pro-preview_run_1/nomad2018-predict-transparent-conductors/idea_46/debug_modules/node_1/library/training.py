import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
)
from library.model import GPIMSDS
from library.data_processing import get_dataloaders

# Set random seeds for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


class Trainer:
    """
    Handles the training, validation, and prediction processes for the GPIMSDS model.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Loss function for regression (MSE on log-transformed targets)
        self.criterion = nn.MSELoss()
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        self.best_val_loss = float("inf")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            atom_feats = batch["atom_features"].to(self.device)
            global_feats = batch["global_features"].to(self.device)
            batch_idx = batch["batch_index"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Log transform targets: log(1 + y) to match metric requirements
            targets_log = torch.log1p(targets)

            self.optimizer.zero_grad()
            outputs = self.model(atom_feats, global_feats, batch_idx)
            loss = self.criterion(outputs, targets_log)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)

        return running_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atom_feats = batch["atom_features"].to(self.device)
                global_feats = batch["global_features"].to(self.device)
                batch_idx = batch["batch_index"].to(self.device)
                targets = batch["targets"].to(self.device)

                # Log transform targets for consistent loss calculation
                targets_log = torch.log1p(targets)

                outputs = self.model(atom_feats, global_feats, batch_idx)
                loss = self.criterion(outputs, targets_log)
                running_loss += loss.item() * targets.size(0)

        return running_loss / len(val_loader.dataset)

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        print("Starting training...")
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Update scheduler based on validation loss
            self.scheduler.step(val_loss)

            # Calculate approx RMSLE (sqrt of MSE on log data)
            val_rmsle = np.sqrt(val_loss)

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss (MSE_Log): {train_loss:.10f} - Val Loss (MSE_Log): {val_loss:.10f} - Val RMSLE: {val_rmsle:.10f}"
            )

            # Checkpointing and Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
                print(f"  New best model saved to {MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val Loss: {self.best_val_loss:.10f}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Generating submission...")
        # Load best model weights
        if os.path.exists(MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: No saved model found. Using current model state.")

        self.model.eval()
        predictions = []
        ids = []

        with torch.no_grad():
            for batch in test_loader:
                atom_feats = batch["atom_features"].to(self.device)
                global_feats = batch["global_features"].to(self.device)
                batch_idx = batch["batch_index"].to(self.device)
                batch_ids = batch["ids"].numpy()

                outputs = self.model(atom_feats, global_feats, batch_idx)

                # Inverse transform: exp(x) - 1 to get original scale
                preds = torch.expm1(outputs).cpu().numpy()

                # Ensure non-negative predictions (physical constraint)
                preds = np.maximum(preds, 0.0)

                predictions.append(preds)
                ids.append(batch_ids)

        predictions = np.concatenate(predictions, axis=0)
        ids = np.concatenate(ids, axis=0)

        # Create submission DataFrame
        sub_df = pd.DataFrame(
            {
                "id": ids,
                "formation_energy_ev_natom": predictions[:, 0],
                "bandgap_energy_ev": predictions[:, 1],
            }
        )

        # Sort by ID as required
        sub_df = sub_df.sort_values("id")

        # Save submission file
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(sub_df.head())


def train_and_predict(load_cached_data=True):
    """
    Orchestrates the data loading, training, and prediction pipeline.
    """
    # Get DataLoaders (handles caching logic internally)
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = GPIMSDS().to(device)

    # Initialize Trainer
    trainer = Trainer(model, device)

    # Execute training
    trainer.fit(train_loader, val_loader)

    # Generate submission
    trainer.predict(test_loader)
