import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import (
    SEED,
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    AUX_LOSS_WEIGHT,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SUBMISSION_DIR,
    IDENTITY_COLUMNS,
    TARGET_COL,
    ID_COL,
)
from library.model import MultiTaskLSTM
from library.metrics import compute_final_metric
from library.data_loader import get_dataloaders


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        model,
        device=DEVICE,
        aux_loss_weight=AUX_LOSS_WEIGHT,
        steps_per_epoch=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.aux_loss_weight = aux_loss_weight

        # Binary Cross Entropy for both tasks (toxicity is binary/fractional, identities are binary)
        self.criterion = nn.BCELoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # One Cycle Policy Scheduler
        self.scheduler = None
        if steps_per_epoch:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=LEARNING_RATE,
                epochs=EPOCHS,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
            )

        # Early stopping tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _dynamic_trim(self, input_ids):
        """Trims the batch to the maximum sequence length in the batch."""
        # input_ids: [batch_size, max_len]
        # Pad token is 0. Find max index where token is not 0.
        mask = input_ids != 0
        max_len = mask.sum(dim=1).max().item()
        # Ensure at least length 1 to avoid errors
        max_len = max(1, max_len)
        return input_ids[:, :max_len]

    def train_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        total_tox_loss = 0.0
        total_aux_loss = 0.0

        for batch in train_loader:
            # Unpack batch
            input_ids = batch["input_ids"].to(self.device)
            target = batch["target"].to(self.device).unsqueeze(1)  # [batch_size, 1]
            aux_target = batch["aux_target"].to(
                self.device
            )  # [batch_size, num_identities]

            # Dynamic Padding: Trim sequence to max length in batch
            input_ids = self._dynamic_trim(input_ids)

            self.optimizer.zero_grad()

            # Forward pass
            tox_pred, aux_pred = self.model(input_ids)

            # Calculate losses
            loss_tox = self.criterion(tox_pred, target)
            loss_aux = self.criterion(aux_pred, aux_target)

            # Weighted sum
            loss = loss_tox + self.aux_loss_weight * loss_aux

            # Backward pass
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # Track metrics (weighted by batch size for accurate average)
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_tox_loss += loss_tox.item() * batch_size
            total_aux_loss += loss_aux.item() * batch_size

        dataset_size = len(train_loader.dataset)
        return {
            "loss": total_loss / dataset_size,
            "tox_loss": total_tox_loss / dataset_size,
            "aux_loss": total_aux_loss / dataset_size,
        }

    def evaluate(self, val_loader):
        """Evaluates the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        total_tox_loss = 0.0

        all_targets = []
        all_preds = []
        all_aux_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                target = batch["target"].to(self.device).unsqueeze(1)
                aux_target = batch["aux_target"].to(self.device)

                # Dynamic Padding
                input_ids = self._dynamic_trim(input_ids)

                tox_pred, aux_pred = self.model(input_ids)

                loss_tox = self.criterion(tox_pred, target)
                loss_aux = self.criterion(aux_pred, aux_target)
                loss = loss_tox + self.aux_loss_weight * loss_aux

                total_loss += loss.item() * input_ids.size(0)
                total_tox_loss += loss_tox.item() * input_ids.size(0)

                # Collect data for metric calculation
                all_targets.extend(target.cpu().numpy().flatten())
                all_preds.extend(tox_pred.cpu().numpy().flatten())
                all_aux_targets.extend(aux_target.cpu().numpy())

        dataset_size = len(val_loader.dataset)
        avg_loss = total_loss / dataset_size
        avg_tox_loss = total_tox_loss / dataset_size

        # Reconstruct DataFrame for metric calculation
        # The validation loader in data_loader.py does not yield IDs, but compute_final_metric
        # only needs targets, predictions, and identity columns.
        val_df = pd.DataFrame({TARGET_COL: all_targets, "prediction": all_preds})

        # Add identity columns back to the DataFrame
        aux_targets_np = np.array(all_aux_targets)
        for i, col in enumerate(IDENTITY_COLUMNS):
            val_df[col] = aux_targets_np[:, i]

        # Compute the competition metric
        final_score, metrics_dict = compute_final_metric(
            val_df, TARGET_COL, "prediction", IDENTITY_COLUMNS
        )

        return {
            "loss": avg_loss,
            "tox_loss": avg_tox_loss,
            "score": final_score,
            "metrics": metrics_dict,
        }

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def predict(self, test_loader):
        """Generates predictions for the test set."""
        self.model.eval()
        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                ids = batch["id"]

                # Dynamic Padding
                input_ids = self._dynamic_trim(input_ids)

                # We only need the toxicity prediction for submission
                tox_pred, _ = self.model(input_ids)

                all_ids.extend(ids.numpy())
                all_preds.extend(tox_pred.cpu().numpy().flatten())

        return pd.DataFrame({ID_COL: all_ids, "prediction": all_preds})


def run_training(load_cached_data=True, debug=False):
    """
    Main execution function.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
        debug (bool): Whether to run in debug mode (smaller dataset).
    """
    set_seed()

    # 1. Load Data
    print("Getting DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Initialize Model
    print("Initializing Model...")
    model = MultiTaskLSTM()

    trainer = Trainer(model)

    print("Starting Training...")
    start_time = time.time()

    # 3. Training Loop
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Train
        train_metrics = trainer.train_epoch(train_loader)

        # Validate
        val_metrics = trainer.evaluate(val_loader)

        epoch_time = time.time() - epoch_start

        # Print Metrics (Full precision as requested)
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.2f}s")
        print(
            f"Train Loss: {train_metrics['loss']} (Tox: {train_metrics['tox_loss']}, Aux: {train_metrics['aux_loss']})"
        )
        print(f"Val Loss: {val_metrics['loss']} (Tox: {val_metrics['tox_loss']})")
        print(f"Val Score: {val_metrics['score']}")

        # Early Stopping based on Toxicity Head Loss
        current_val_tox_loss = val_metrics["tox_loss"]

        if current_val_tox_loss < trainer.best_val_loss:
            print(
                f"Validation Toxicity Loss improved from {trainer.best_val_loss} to {current_val_tox_loss}. Saving model..."
            )
            trainer.best_val_loss = current_val_tox_loss
            trainer.save_model(MODEL_SAVE_PATH)
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1
            print(
                f"No improvement in Toxicity Loss. Patience: {trainer.patience_counter}/{PATIENCE}"
            )

        if trainer.patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")

    # 4. Generate Submission
    print("Loading best model for prediction...")
    trainer.load_model(MODEL_SAVE_PATH)

    print("Generating predictions on test set...")
    submission_df = trainer.predict(test_loader)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved.")
