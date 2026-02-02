import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.data import get_dataloaders
from library.model import MCPDSModel


def set_seed(seed):
    """
    Sets random seeds for reproducibility.
    """
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
    Manages the training, validation, and prediction lifecycle of the MC-PDS model.
    """

    def __init__(self, model):
        self.model = model.to(Config.DEVICE)
        self.criterion = nn.MSELoss()

        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler to reduce learning rate when validation loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Move data to device
            atomic_feats = batch["atomic_features"].to(Config.DEVICE)
            global_feats = batch["global_features"].to(Config.DEVICE)
            mask = batch["mask"].to(Config.DEVICE)
            targets = batch["targets"].to(Config.DEVICE)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(atomic_feats, global_feats, mask)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atomic_feats = batch["atomic_features"].to(Config.DEVICE)
                global_feats = batch["global_features"].to(Config.DEVICE)
                mask = batch["mask"].to(Config.DEVICE)
                targets = batch["targets"].to(Config.DEVICE)

                outputs = self.model(atomic_feats, global_feats, mask)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * targets.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss

    def fit(
        self, train_loader, val_loader, epochs=Config.EPOCHS, patience=Config.PATIENCE
    ):
        """
        Runs the full training loop with early stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs with patience {patience}...")

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Step the scheduler
            self.scheduler.step(val_loss)

            duration = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Time: {duration:.2f}s"
            )

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"  -> New best model saved (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")

    def generate_submission(self, test_loader, output_path):
        """
        Generates predictions for the test set and saves to CSV.
        Loads the best model state before prediction.
        """
        print(f"Loading best model from {Config.MODEL_PATH}...")
        self.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        )
        self.model.eval()

        all_ids = []
        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                atomic_feats = batch["atomic_features"].to(Config.DEVICE)
                global_feats = batch["global_features"].to(Config.DEVICE)
                mask = batch["mask"].to(Config.DEVICE)
                ids = batch["ids"]

                # Forward pass
                outputs = self.model(atomic_feats, global_feats, mask)

                # Inverse transform: targets were log1p transformed
                # pred_original = exp(pred_log) - 1
                preds_original = torch.expm1(outputs)

                all_ids.extend(ids.numpy())
                all_preds.extend(preds_original.cpu().numpy())

        # Create DataFrame
        preds_array = np.array(all_preds)
        submission_df = pd.DataFrame(
            {
                "id": all_ids,
                "formation_energy_ev_natom": preds_array[:, 0],
                "bandgap_energy_ev": preds_array[:, 1],
            }
        )

        # Save
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def train_and_evaluate():
    """
    Main function to coordinate data loading, training, and submission generation.
    """
    # 1. Setup
    set_seed(Config.SEED)
    Config.print_config()

    # 2. Data Loading
    # Using cached data if available for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = MCPDSModel()

    # 4. Training
    trainer = Trainer(model)
    trainer.fit(
        train_loader, val_loader, epochs=Config.EPOCHS, patience=Config.PATIENCE
    )

    # 5. Submission
    trainer.generate_submission(test_loader, Config.SUBMISSION_PATH)
