import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.model import AMSP_DS_Net
from library.data import get_data_loaders
from library.utils import inverse_log_transform


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Handles training, validation, and inference for the AMSP-DS model.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Move data to device
            atomic_features = batch["atomic_features"].to(self.device)
            batch_indices = batch["batch_indices"].to(self.device)
            global_features = batch["global_features"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(atomic_features, batch_indices, global_features)

            # Compute loss (MSE on log-transformed targets)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        # Track separate losses for formation energy and bandgap
        total_loss_formation = 0.0
        total_loss_bandgap = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atomic_features = batch["atomic_features"].to(self.device)
                batch_indices = batch["batch_indices"].to(self.device)
                global_features = batch["global_features"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(atomic_features, batch_indices, global_features)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)

                # Per-column MSE (squared error sum)
                errors = (outputs - targets) ** 2
                total_loss_formation += errors[:, 0].sum().item()
                total_loss_bandgap += errors[:, 1].sum().item()

        N = len(val_loader.dataset)
        avg_loss = total_loss / N
        rmsle_formation = np.sqrt(total_loss_formation / N)
        rmsle_bandgap = np.sqrt(total_loss_bandgap / N)

        return avg_loss, rmsle_formation, rmsle_bandgap

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
    ):
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_rmsle_form, val_rmsle_gap = self.validate(val_loader)

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val RMSLE (Form): {val_rmsle_form:.6f} | "
                f"Val RMSLE (Gap): {val_rmsle_gap:.6f}"
            )

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT)
                # print(f"  -> Model saved (Best Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")

    def predict(self, test_loader):
        self.model.eval()
        # Load best model state
        if os.path.exists(Config.MODEL_CHECKPOINT):
            self.model.load_state_dict(
                torch.load(Config.MODEL_CHECKPOINT, map_location=self.device)
            )
            print("Loaded best model for prediction.")
        else:
            print("Warning: No checkpoint found, using current model state.")

        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                atomic_features = batch["atomic_features"].to(self.device)
                batch_indices = batch["batch_indices"].to(self.device)
                global_features = batch["global_features"].to(self.device)
                ids = batch["ids"]

                # Forward pass (log scale)
                outputs_log = self.model(
                    atomic_features, batch_indices, global_features
                )

                # Inverse transform to original scale
                outputs_original = inverse_log_transform(outputs_log)

                # Clip negative values to 0 (physics constraint)
                outputs_original = torch.clamp(outputs_original, min=0.0)

                all_preds.append(outputs_original.cpu().numpy())
                all_ids.extend(ids)

        return np.array(all_ids), np.vstack(all_preds)


def run_training():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE, load_cached=True, num_workers=2
    )

    # 3. Model Initialization
    # We need to determine input dimensions from the data
    # Get a sample batch to check dimensions
    sample_batch = next(iter(train_loader))
    atom_dim = sample_batch["atomic_features"].shape[1]
    global_dim = sample_batch["global_features"].shape[1]

    print(f"Atomic Feature Dim: {atom_dim}")
    print(f"Global Feature Dim: {global_dim}")

    model = AMSP_DS_Net(
        atom_input_dim=atom_dim,
        global_input_dim=global_dim,
        atomic_hidden_dim=Config.ATOMIC_HIDDEN_DIM,
        atomic_layers=Config.ATOMIC_LAYERS,
        global_hidden_dim=Config.GLOBAL_HIDDEN_DIM,
        global_layers=Config.GLOBAL_LAYERS,
        fusion_hidden_dim=Config.FUSION_HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # 4. Training
    trainer = Trainer(model)
    trainer.fit(train_loader, val_loader)

    # 5. Prediction
    print("Generating predictions...")
    ids, preds = trainer.predict(test_loader)

    # 6. Submission Generation
    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )

    # Sort by ID to match sample submission structure (good practice)
    submission_df.sort_values("id", inplace=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print head
    print("\nSubmission Head:")
    print(submission_df.head())


if __name__ == "__main__":
    run_training()
