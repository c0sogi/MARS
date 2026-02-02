import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.loss import WeightedL1Loss, competition_metric
from library.model import DeepBiLSTM
from library.dataset import get_ventilator_dataset


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the
    Ventilator Pressure Prediction model.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = DeepBiLSTM(
            input_dim=Config.INPUT_DIM,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            glu_width=Config.INJECTION_GLU_WIDTH,
            dropout=Config.DROPOUT,
        ).to(self.device)

        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing (Stretched Horizon)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.COSINE_T_MAX, eta_min=1e-6
        )

        # Loss Function
        self.criterion = WeightedL1Loss()

        # Best score tracking
        self.best_val_mae = float("inf")

    def train_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0

        for batch_X, batch_y, batch_u_out in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_u_out = batch_u_out.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(batch_X)

            # Compute Loss
            loss = self.criterion(preds, batch_y, batch_u_out)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """Runs validation and calculates the competition metric."""
        self.model.eval()
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_X, batch_y, batch_u_out in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_u_out = batch_u_out.to(self.device)

                # Forward pass
                preds = self.model(batch_X)

                # Calculate Metric (Inspiratory Phase MAE)
                mae = competition_metric(preds, batch_y, batch_u_out)

                # competition_metric returns a float item, not a tensor
                total_mae += mae
                num_batches += 1

        return total_mae / num_batches if num_batches > 0 else 0.0

    def fit(self, train_loader, val_loader):
        """
        Executes the training pipeline with Early Stopping and Scheduler.
        """
        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs...")
        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_mae = self.validate(val_loader)

            # Scheduler Step
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Time: {elapsed:.2f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val MAE: {val_mae:.8f}"
            )

            # Checkpoint & Early Stopping
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"  >>> New Best Model Saved! (MAE: {val_mae:.8f})")
            else:
                patience_counter += 1
                print(
                    f"  >>> No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Starting inference...")

        # Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        all_preds = []

        with torch.no_grad():
            for batch_X, _, _ in test_loader:
                batch_X = batch_X.to(self.device)

                # Forward pass
                preds = self.model(batch_X)

                # Move to CPU and flatten
                # Preds shape: (Batch, Seq_Len) -> (Batch * Seq_Len)
                preds_flat = preds.cpu().numpy().flatten()
                all_preds.append(preds_flat)

        # Concatenate all batches
        final_predictions = np.concatenate(all_preds)

        # ---------------------------------------------------------
        # Align with IDs
        # ---------------------------------------------------------
        # The model processes data sorted by breath_id and time_step.
        # We must load the test metadata and sort it identically to ensure
        # the flattened predictions match the correct IDs.
        print("Loading test metadata for ID alignment...")
        df_test_meta = pd.read_csv(Config.TEST_METADATA)

        # Sort to match data processing order
        df_test_meta = df_test_meta.sort_values([Config.BREATH_ID_COL, "time_step"])

        # Verify lengths match
        if len(final_predictions) != len(df_test_meta):
            raise ValueError(
                f"Prediction count ({len(final_predictions)}) does not match "
                f"metadata row count ({len(df_test_meta)})."
            )

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                Config.ID_COL: df_test_meta[Config.ID_COL].values,
                Config.TARGET_COL: final_predictions,
            }
        )

        # Sort by ID just in case submission requires specific order (usually ID ascending)
        submission = submission.sort_values(Config.ID_COL)

        # Save
        print(f"Saving submission to {Config.SUBMISSION_FILE}...")
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print("Submission saved successfully.")


def run_training(debug=Config.DEBUG):
    """
    Main entry point to execute the training pipeline.
    """
    set_seed(Config.SEED)

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = get_ventilator_dataset("train", debug=debug)
    val_dataset = get_ventilator_dataset("val", debug=debug)
    test_dataset = get_ventilator_dataset("test", debug=debug)

    # 2. Create DataLoaders
    # Pin memory for faster host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Trainer
    trainer = Trainer()

    # 4. Train
    trainer.fit(train_loader, val_loader)

    # 5. Predict
    trainer.predict(test_loader)
