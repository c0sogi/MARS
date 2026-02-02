import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from library.config import Config
from library.data import MaterialDataset, collate_batch
from library.model import ChemicallyWeightedDeepSets


def set_seed(seed):
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
    Manages the training lifecycle of the Chemically-Weighted Deep Sets model.
    """

    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        # Loss function: MSE on log-transformed targets
        self.criterion = nn.MSELoss()

        # Optimizer: AdamW with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler: Reduce LR on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )

    def train_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Move batch to device
            atomic_features = batch["atomic_features"].to(self.device)
            atomic_mask = batch["atomic_mask"].to(self.device)
            global_features = batch["global_features"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(atomic_features, atomic_mask, global_features)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate loss (weighted by batch size)
            total_loss += loss.item() * targets.size(0)

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        """Evaluates the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        total_mse_1 = 0.0
        total_mse_2 = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atomic_features = batch["atomic_features"].to(self.device)
                atomic_mask = batch["atomic_mask"].to(self.device)
                global_features = batch["global_features"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(atomic_features, atomic_mask, global_features)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * targets.size(0)

                # Calculate column-wise squared errors for reporting
                # Since targets are log(1+y) and outputs are predicted log(1+y),
                # MSE here is equivalent to MSLE on the original scale.
                # RMSLE is sqrt(MSE).
                squared_errors = (outputs - targets) ** 2
                total_mse_1 += torch.sum(squared_errors[:, 0]).item()
                total_mse_2 += torch.sum(squared_errors[:, 1]).item()

        avg_loss = total_loss / len(val_loader.dataset)
        rmsle_1 = np.sqrt(total_mse_1 / len(val_loader.dataset))
        rmsle_2 = np.sqrt(total_mse_2 / len(val_loader.dataset))

        return avg_loss, rmsle_1, rmsle_2

    def fit(self, train_loader, val_loader, epochs, patience, checkpoint_path):
        """Runs the full training loop with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_rmsle_1, val_rmsle_2 = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"Val RMSLE Form: {val_rmsle_1:.8f} | "
                f"Val RMSLE Gap: {val_rmsle_2:.8f}"
            )

            # Checkpoint and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  -> New best model saved (Val Loss: {val_loss:.8f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best Val Loss: {best_val_loss:.8f}")

    def predict(self, test_loader):
        """Generates predictions for the test set."""
        self.model.eval()
        predictions = []
        ids = []

        with torch.no_grad():
            for batch in test_loader:
                atomic_features = batch["atomic_features"].to(self.device)
                atomic_mask = batch["atomic_mask"].to(self.device)
                global_features = batch["global_features"].to(self.device)
                batch_ids = batch["ids"]

                outputs = self.model(atomic_features, atomic_mask, global_features)

                predictions.append(outputs.cpu().numpy())
                ids.append(batch_ids.numpy())

        return np.concatenate(predictions, axis=0), np.concatenate(ids, axis=0)


def train_model(load_cached_data=True, sample_size=None):
    """
    Main function to train the model.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed data from disk.
        sample_size (int, optional): Limit dataset size for debugging.
    """
    set_seed(Config.SEED)

    # Load Datasets
    print("Initializing datasets...")
    train_dataset = MaterialDataset(
        mode="train", load_cached_data=load_cached_data, sample_size=sample_size
    )
    val_dataset = MaterialDataset(
        mode="val", load_cached_data=load_cached_data, sample_size=sample_size
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = ChemicallyWeightedDeepSets()

    # Initialize Trainer
    trainer = Trainer(
        model,
        Config.DEVICE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Start Training
    trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        checkpoint_path=Config.MODEL_CHECKPOINT,
    )


def generate_submission(load_cached_data=True):
    """
    Generates the submission file using the best trained model.
    """
    set_seed(Config.SEED)

    # Load Test Dataset
    print("Initializing test dataset...")
    test_dataset = MaterialDataset(mode="test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = ChemicallyWeightedDeepSets()
    if os.path.exists(Config.MODEL_CHECKPOINT):
        model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT, map_location=Config.DEVICE)
        )
        print(f"Loaded model checkpoint from {Config.MODEL_CHECKPOINT}")
    else:
        print(
            f"Warning: Checkpoint not found at {Config.MODEL_CHECKPOINT}. Using untrained model."
        )

    trainer = Trainer(model, Config.DEVICE)

    # Predict
    print("Generating predictions...")
    preds_log, ids = trainer.predict(test_loader)

    # Inverse Transform: exp(x) - 1
    # The model predicts log(1 + y), so we reverse this to get y.
    preds_original = np.expm1(preds_log)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds_original[:, 0],
            "bandgap_energy_ev": preds_original[:, 1],
        }
    )

    # Sort by ID to match sample submission format
    submission_df.sort_values("id", inplace=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
