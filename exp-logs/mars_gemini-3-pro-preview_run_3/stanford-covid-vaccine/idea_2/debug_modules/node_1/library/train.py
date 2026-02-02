import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_dataloaders
from library.model import HybridGNN


class Trainer:
    """
    Manages the training and validation of the RNA degradation model.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, device, save_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.best_val_loss = float("inf")

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # Output shape: (Batch_Size, Seq_Len, Num_Targets)
            preds = self.model(batch)

            # Prepare targets
            # PyG collates targets by concatenating them along the first dimension.
            # batch.y shape: (Batch_Size * Seq_Len, Num_Targets)
            targets = batch.y

            # Reshape targets to match predictions: (Batch_Size, Seq_Len, Num_Targets)
            batch_size = preds.size(0)
            seq_len = Config.SEQ_LENGTH

            # Ensure targets are reshaped correctly
            if targets.size(0) == batch_size * seq_len:
                targets = targets.view(batch_size, seq_len, -1)
            else:
                # Fallback or error if shapes don't align (should not happen with correct data)
                raise ValueError(
                    f"Target shape mismatch. Expected {batch_size*seq_len} rows, got {targets.size(0)}"
                )

            # Calculate Loss
            # mcrmse_loss handles slicing internally based on Config.SEQ_SCORED
            loss = mcrmse_loss(preds, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def validate(self):
        """
        Evaluates the model on the validation set.
        Accumulates predictions to calculate the global MCRMSE.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                preds = self.model(batch)
                targets = batch.y

                # Reshape targets
                batch_size = preds.size(0)
                seq_len = Config.SEQ_LENGTH
                if targets.size(0) == batch_size * seq_len:
                    targets = targets.view(batch_size, seq_len, -1)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        if not all_preds:
            return float("inf")

        # Concatenate all batches
        full_preds = torch.cat(all_preds, dim=0)
        full_targets = torch.cat(all_targets, dim=0)

        # Calculate metric on the full validation set
        val_loss = mcrmse_loss(full_preds, full_targets)
        return val_loss.item()

    def fit(self, num_epochs, patience):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training on {self.device}...")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                # Save model
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.save_path)
                print(f"New best model saved to {self.save_path}")
            else:
                patience_counter += 1
                print(f"EarlyStopping counter: {patience_counter} out of {patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.EARLY_STOPPING_PATIENCE,
    load_cached_data=True,
):
    """
    Main function to setup and run the training process.
    """
    # 1. Set Seed for Reproducibility
    set_seed()

    # 2. Prepare DataLoaders
    # We only need train and val loaders here
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 3. Initialize Model
    device = torch.device(Config.DEVICE)
    model = HybridGNN().to(device)

    # 4. Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Setup Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 6. Start Training
    trainer.fit(num_epochs, patience)

    return trainer.best_val_loss
