import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.model import SCC_WDS_Net
from library.data import get_train_val_loaders, get_test_loader
from library.utils import set_seed, inverse_log_transform


class Engine:
    def __init__(self, device=Config.DEVICE, seed=Config.SEED):
        """
        Initializes the Engine with device and seed settings.
        """
        self.device = device
        set_seed(seed)

    def train_epoch(self, model, loader, optimizer, criterion):
        """
        Runs one epoch of training.
        """
        model.train()
        total_loss = 0.0
        num_samples = 0

        for batch in loader:
            # Move data to device
            atomic_feats = batch["atomic_features"].to(self.device)
            global_feats = batch["global_features"].to(self.device)
            batch_idx = batch["batch_index"].to(self.device)
            targets = batch["target"].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(atomic_feats, global_feats, batch_idx)

            # Loss calculation
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss (MSE is averaged over batch, so multiply by batch size)
            total_loss += loss.item() * targets.size(0)
            num_samples += targets.size(0)

        return total_loss / num_samples

    def validate(self, model, loader, criterion):
        """
        Evaluates the model on the validation set.
        """
        model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in loader:
                atomic_feats = batch["atomic_features"].to(self.device)
                global_feats = batch["global_features"].to(self.device)
                batch_idx = batch["batch_index"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = model(atomic_feats, global_feats, batch_idx)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * targets.size(0)
                num_samples += targets.size(0)

        return total_loss / num_samples

    def run_training(
        self,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        load_cached_data=True,
    ):
        """
        Orchestrates the full training process with early stopping.
        """
        print(f"Initializing training on {self.device}...")

        # Get DataLoaders
        train_loader, val_loader = get_train_val_loaders(
            batch_size=batch_size, load_cached_data=load_cached_data
        )

        # Initialize Model
        model = SCC_WDS_Net().to(self.device)

        # Optimizer and Loss
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
        )
        criterion = nn.MSELoss()

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        patience_counter = 0

        print("Starting training loop...")
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss = self.validate(model, val_loader, criterion)

            # Step scheduler
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Ensure directory exists before saving
                os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), Config.MODEL_PATH)
                print(f"  -> New best model saved! (Val Loss: {val_loss:.10f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Validation Loss: {best_val_loss:.10f}")
        return best_val_loss

    def predict(self, batch_size=Config.BATCH_SIZE, load_cached_data=True):
        """
        Generates predictions for the test set using the best trained model.
        """
        print("Starting inference...")

        # Get Test Loader
        test_loader = get_test_loader(
            batch_size=batch_size, load_cached_data=load_cached_data
        )

        # Load Model
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {Config.MODEL_PATH}. Please train the model first."
            )

        model = SCC_WDS_Net().to(self.device)
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=self.device))
        model.eval()

        all_preds = []
        all_ids = []

        with torch.no_grad():
            for batch in test_loader:
                atomic_feats = batch["atomic_features"].to(self.device)
                global_feats = batch["global_features"].to(self.device)
                batch_idx = batch["batch_index"].to(self.device)
                ids = batch["id"]

                outputs = model(atomic_feats, global_feats, batch_idx)

                # Inverse transform log-targets
                preds = inverse_log_transform(outputs.cpu().numpy())

                all_preds.append(preds)
                all_ids.append(ids.numpy())

        # Concatenate results
        predictions = np.concatenate(all_preds, axis=0)
        ids = np.concatenate(all_ids, axis=0)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {
                "id": ids,
                "formation_energy_ev_natom": predictions[:, 0],
                "bandgap_energy_ev": predictions[:, 1],
            }
        )

        # Sort by ID to ensure correct order
        submission_df = submission_df.sort_values("id")

        # Save to CSV
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

        return submission_df
