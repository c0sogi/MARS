import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import (
    seed_everything,
    mcrmse_metric,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import get_dataloaders
from library.model import HybridCNNBiGRU
from library.loss import MaskedMSELoss


class Trainer:
    """
    Manages the training, validation, and inference process for the RNA degradation model.
    """

    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, scheduler, device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_score = float("inf")

        # Indices of the targets that are actually scored in the competition
        # 0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C
        # Based on Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        self.scored_target_indices = [0, 1, 3]

    def train_epoch(self, epoch):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            # Move data to device
            sequence = batch["sequence"].to(self.device)
            structure = batch["structure"].to(self.device)
            loop_type = batch["predicted_loop_type"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(sequence, structure, loop_type)

            # Compute loss
            loss = self.criterion(outputs, targets, mask)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                sequence = batch["sequence"].to(self.device)
                structure = batch["structure"].to(self.device)
                loop_type = batch["predicted_loop_type"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(sequence, structure, loop_type)
                loss = self.criterion(outputs, targets, mask)
                running_loss += loss.item()

                # For MCRMSE metric calculation:
                # 1. Select only the scored sequence positions (first 68)
                # 2. Keep all target columns initially (filtering happens in metric function)
                valid_preds = outputs[:, : Config.SEQ_SCORED, :]
                valid_targets = targets[:, : Config.SEQ_SCORED, :]

                all_preds.append(valid_preds.cpu().numpy())
                all_targets.append(valid_targets.cpu().numpy())

        # Concatenate batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Metric
        # Pass scored_indices to calculate MCRMSE only on the 3 scored target types
        metric = mcrmse_metric(
            all_targets, all_preds, scored_indices=self.scored_target_indices
        )

        avg_loss = running_loss / len(self.val_loader)
        return avg_loss, metric

    def fit(self, epochs, patience):
        """Main training loop with Early Stopping."""
        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_metric = self.validate()

            # Scheduler step (ReduceLROnPlateau monitors the validation metric)
            self.scheduler.step(val_metric)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MCRMSE: {val_metric} | "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpointing
            if val_metric < self.best_score:
                self.best_score = val_metric
                patience_counter = 0

                # Save best model
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_score": self.best_score,
                    },
                    is_best=True,
                    filename=os.path.join(Config.WORKING_DIR, "checkpoint.pth"),
                )

                print(f"New best model found! Score: {self.best_score}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

    def predict(self, test_loader):
        """Generates predictions for the test set."""
        self.model.eval()
        all_ids = []
        all_preds = []

        print("Generating predictions on test set...")
        with torch.no_grad():
            for batch in test_loader:
                ids = batch["id"]
                sequence = batch["sequence"].to(self.device)
                structure = batch["structure"].to(self.device)
                loop_type = batch["predicted_loop_type"].to(self.device)

                # Forward pass
                outputs = self.model(sequence, structure, loop_type)

                # Store predictions: (Batch, 107, 5)
                # We output predictions for all positions as required by the submission format
                all_preds.append(outputs.cpu().numpy())
                all_ids.extend(ids)

        return all_ids, np.concatenate(all_preds, axis=0)


def generate_submission(ids, preds, output_path):
    """
    Formats predictions into the competition CSV format.

    Args:
        ids (list): List of sample IDs.
        preds (np.array): Prediction tensor of shape (N_samples, 107, 5).
        output_path (str): Path to save the CSV file.
    """
    target_cols = Config.TARGET_COLS

    id_seqpos_list = []
    flat_preds = []

    seq_len = preds.shape[1]  # 107

    # Flatten predictions to one row per position
    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 5)

        for pos in range(seq_len):
            id_seqpos_list.append(f"{sample_id}_{pos}")
            flat_preds.append(sample_preds[pos])

    flat_preds = np.array(flat_preds)

    # Create DataFrame
    submission_df = pd.DataFrame(flat_preds, columns=target_cols)
    submission_df.insert(0, "id_seqpos", id_seqpos_list)

    # Save
    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)
    print("Submission saved successfully.")


def run_training():
    """
    Main execution function.
    Sets up the environment, loads data, trains the model, and generates submission.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Use load_cached_data=True to leverage pre-processed .npz files
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = HybridCNNBiGRU().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = MaskedMSELoss()

    # 5. Trainer Initialization
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # 6. Training Loop
    trainer.fit(epochs=Config.EPOCHS, patience=Config.PATIENCE)

    # 7. Inference
    print("Loading best model for inference...")
    load_checkpoint(model, filename=Config.MODEL_SAVE_PATH)

    test_ids, test_preds = trainer.predict(test_loader)

    # 8. Submission
    generate_submission(test_ids, test_preds, Config.SUBMISSION_PATH)


# Execute the training pipeline
run_training()
