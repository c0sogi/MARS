import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, seed_everything
from library.data_processing import DataHandler
from library.model import PAWDS, collate_fn

logger = get_logger("train")


class Trainer:
    """
    Manages the training lifecycle of the PA-WDS model.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_path = Config.MODEL_SAVE_PATH

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            atomic_x = batch["atomic_features"].to(self.device)
            global_x = batch["global_features"].to(self.device)
            mask = batch["mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(atomic_x, global_x, mask)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * atomic_x.size(0)

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
                atomic_x = batch["atomic_features"].to(self.device)
                global_x = batch["global_features"].to(self.device)
                mask = batch["mask"].to(self.device)
                targets = batch["targets"].to(self.device)

                outputs = self.model(atomic_x, global_x, mask)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * atomic_x.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss

    def fit(self, train_loader, val_loader, num_epochs=Config.NUM_EPOCHS):
        """
        Runs the full training loop with early stopping.
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(
                    f"Epoch {epoch+1}: Validation loss improved to {val_loss}. Model saved."
                )
            else:
                self.patience_counter += 1

            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Time: {elapsed:.2f}s"
            )

            if self.patience_counter >= Config.PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

        logger.info(f"Training complete. Best Validation Loss: {self.best_val_loss}")


def run():
    """
    Main execution function:
    1. Setup and Data Loading
    2. Model Initialization
    3. Training
    4. Inference and Submission Generation
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    data_handler = DataHandler()
    # The DataHandler handles caching internally.
    train_dataset, val_dataset, test_dataset = data_handler.get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = PAWDS().to(device)

    # 4. Training
    trainer = Trainer(model, device)
    trainer.fit(train_loader, val_loader)

    # 5. Inference
    logger.info("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()

    predictions = []
    ids = []

    logger.info("Generating predictions on test set...")
    with torch.no_grad():
        for batch in test_loader:
            atomic_x = batch["atomic_features"].to(device)
            global_x = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["ids"]

            outputs = model(atomic_x, global_x, mask)

            # Inverse transform: log(1+y) -> exp(x) - 1
            # Ensure non-negative predictions by clamping before exp if necessary,
            # though exp(x) is always positive.
            preds_original_scale = torch.expm1(outputs).cpu().numpy()

            predictions.append(preds_original_scale)
            ids.extend(batch_ids)

    predictions = np.vstack(predictions)

    # 6. Submission
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Ensure columns are in correct order
    submission_df = submission_df[
        ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    ]

    # Sort by ID to be safe
    submission_df = submission_df.sort_values("id")

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
