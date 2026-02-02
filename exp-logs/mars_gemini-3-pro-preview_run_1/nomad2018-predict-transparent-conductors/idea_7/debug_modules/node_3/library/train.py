import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import MaterialDataset, collate_batch
from library.model import SIRDSModel, predict as model_predict


class Trainer:
    """
    Manages the training lifecycle of the SI-RDS model.
    """

    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
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
            min_lr=config.SCHEDULER_MIN_LR,
        )
        self.best_val_loss = float("inf")

    def train_one_epoch(self, train_loader):
        """
        Trains the model for one epoch.
        Returns the average training loss (MSE).
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            # Move data to device
            atomic = batch["atomic_features"].to(self.device)
            global_f = batch["global_features"].to(self.device)
            sym = batch["symmetry"].to(self.device)
            mask = batch["mask"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(atomic, global_f, sym, mask)
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()

            # Accumulate loss weighted by batch size
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / n_samples

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns the average MSE loss and RMSLE.
        """
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                atomic = batch["atomic_features"].to(self.device)
                global_f = batch["global_features"].to(self.device)
                sym = batch["symmetry"].to(self.device)
                mask = batch["mask"].to(self.device)
                targets = batch["targets"].to(self.device)

                preds = self.model(atomic, global_f, sym, mask)
                loss = self.criterion(preds, targets)

                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                n_samples += batch_size

        mse = total_loss / n_samples
        # Since targets are log1p transformed, RMSE on log data is RMSLE on original data
        rmsle = np.sqrt(mse)
        return mse, rmsle

    def fit(self, train_loader, val_loader):
        """
        Runs the full training loop with early stopping and scheduling.
        """
        print(f"Starting training on {self.device}...")
        patience_counter = 0

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_one_epoch(train_loader)
            val_mse, val_rmsle = self.evaluate(val_loader)

            # Update learning rate
            self.scheduler.step(val_mse)

            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} - Train Loss: {train_loss} - Val MSE: {val_mse} - Val RMSLE: {val_rmsle}"
            )

            # Checkpointing and Early Stopping
            if val_mse < self.best_val_loss:
                self.best_val_loss = val_mse
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Best Validation MSE: {self.best_val_loss}")


def run_training(debug=False):
    """
    Sets up datasets, model, and trainer, then executes training.
    """
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    debug_size = 100 if debug else None

    # Initialize Datasets
    # Train dataset initialization will fit and save scalers
    print("Initializing Training Dataset...")
    train_dataset = MaterialDataset(
        metadata_path=config.TRAIN_CSV,
        geometry_dir=config.GEOMETRY_DIR,
        cache_path=config.TRAIN_CACHE,
        load_cached_data=True,
        debug_sample_size=debug_size,
        mode="train",
    )

    print("Initializing Validation Dataset...")
    val_dataset = MaterialDataset(
        metadata_path=config.VAL_CSV,
        geometry_dir=config.GEOMETRY_DIR,
        cache_path=config.VAL_CACHE,
        load_cached_data=True,
        debug_sample_size=debug_size,
        mode="val",
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = SIRDSModel(config)

    # Trainer
    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)

    return trainer.model


def generate_submission(model=None, device=None):
    """
    Generates predictions for the test set and saves the submission file.
    """
    config = Config()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model if not provided
    if model is None:
        model = SIRDSModel(config)
        if os.path.exists(config.MODEL_PATH):
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        else:
            print(
                "Warning: No trained model found at checkpoint path. Using untrained model."
            )

    model.to(device)

    print("Initializing Test Dataset...")
    # Test dataset will load scalers saved during training
    test_dataset = MaterialDataset(
        metadata_path=config.TEST_CSV,
        geometry_dir=config.GEOMETRY_DIR,
        cache_path=config.TEST_CACHE,
        load_cached_data=True,
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=2,
        pin_memory=True,
    )

    print("Generating predictions...")
    # model_predict handles the inverse transformation (expm1)
    preds, ids = model_predict(model, test_loader, device)

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )

    # Save
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
