import os
import torch
import torch.nn as nn
import numpy as np
from library import config, utils, model, data


class Trainer:
    """
    Trainer class for managing the training and validation of the LP-RA-CGN model.
    """

    def __init__(self, model, train_loader, val_loader, target_scaler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_scaler = target_scaler
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
        )

    def train_epoch(self):
        """
        Executes one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            preds = self.model(batch)

            targets = batch.y
            # Standardize targets for training stability
            if self.target_scaler is not None:
                targets = self.target_scaler.transform(targets)

            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(self.train_loader.dataset)

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns average loss (scaled) and RMSLE (original scale).
        """
        self.model.eval()
        total_loss = 0.0
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                preds = self.model(batch)
                targets = batch.y

                # Calculate loss on scaled targets for scheduler consistency
                if self.target_scaler is not None:
                    targets_scaled = self.target_scaler.transform(targets)
                    loss = self.criterion(preds, targets_scaled)
                else:
                    loss = self.criterion(preds, targets)

                total_loss += loss.item() * batch.num_graphs

                # Inverse transform predictions for metric calculation
                if self.target_scaler is not None:
                    preds_original = self.target_scaler.inverse_transform(preds)
                else:
                    preds_original = preds

                preds_list.append(preds_original.cpu())
                targets_list.append(batch.y.cpu())

        avg_loss = total_loss / len(self.val_loader.dataset)

        all_preds = torch.cat(preds_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)

        # Compute RMSLE on original scale
        rmsle = utils.compute_rmsle(all_preds, all_targets)

        return avg_loss, rmsle

    def fit(self, epochs=config.MAX_EPOCHS):
        """
        Runs the full training loop with early stopping and checkpointing.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_rmsle = self.validate()

            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val RMSLE: {val_rmsle}"
            )

            self.scheduler.step(val_loss)

            # Checkpointing based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                )
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val Loss: {best_val_loss}")

        # Load best model state
        best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model checkpoint.")


def run_training(
    load_cached_data=True, batch_size=config.BATCH_SIZE, epochs=config.MAX_EPOCHS
):
    """
    Orchestrates the training pipeline: data loading, model init, and training.
    """
    # Ensure reproducibility
    utils.set_seed(config.SEED)

    # Load data
    train_loader, val_loader, test_loader, target_scaler = data.get_dataloaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    # Initialize model
    # Node input dim 100 covers all elements in the dataset
    model_instance = model.LP_RA_CGN(node_input_dim=100).to(config.DEVICE)

    # Initialize trainer
    trainer = Trainer(
        model_instance, train_loader, val_loader, target_scaler, config.DEVICE
    )

    # Execute training
    trainer.fit(epochs=epochs)

    return trainer.model, test_loader, target_scaler


def make_submission(model_instance, test_loader, target_scaler):
    """
    Wrapper to generate submission file using the trained model.
    """
    model.generate_submission(model_instance, test_loader, target_scaler, config.DEVICE)
