import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import (
    WORKING_DIR,
    MODEL_CHECKPOINT_PATH,
    SUBMISSION_OUTPUT_PATH,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    DEVICE,
    SEED,
)
from library.utils import meters_to_degrees_diff
from library.model import SkyMotionModel
from library.data_loader import get_train_val_loaders, get_test_loader


# Set random seeds for reproducibility
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, model):
        self.model = model.to(DEVICE)
        self.criterion = nn.L1Loss()  # Mean Absolute Error
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )
        self.best_val_loss = float("inf")

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for traj, sky, target in train_loader:
            traj = traj.to(DEVICE)
            sky = sky.to(DEVICE)
            target = target.to(DEVICE)

            self.optimizer.zero_grad()

            outputs = self.model(traj, sky)
            loss = self.criterion(outputs, target)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * traj.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for traj, sky, target in val_loader:
                traj = traj.to(DEVICE)
                sky = sky.to(DEVICE)
                target = target.to(DEVICE)

                outputs = self.model(traj, sky)
                loss = self.criterion(outputs, target)

                running_loss += loss.item() * traj.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss

    def fit(self, train_loader, val_loader):
        patience_counter = 0

        print(f"Starting training on device: {DEVICE}")

        for epoch in range(NUM_EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.10f} - "
                f"Val Loss: {val_loss:.10f}"
            )

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                print(
                    f"Validation loss improved from {self.best_val_loss:.10f} to {val_loss:.10f}. Saving model..."
                )
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), MODEL_CHECKPOINT_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}"
                )

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")

    def predict(self, test_loader):
        # Load best model
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found at {MODEL_CHECKPOINT_PATH}"
            )

        print(f"Loading best model from {MODEL_CHECKPOINT_PATH}...")
        self.model.load_state_dict(
            torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
        )
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for traj, sky in tqdm(test_loader, desc="Predicting"):
                traj = traj.to(DEVICE)
                sky = sky.to(DEVICE)

                outputs = self.model(traj, sky)
                predictions.append(outputs.cpu().numpy())

        return np.vstack(predictions)


def train_model(load_cached_data=True):
    set_seed()

    # Get DataLoaders
    train_loader, val_loader = get_train_val_loaders(load_cached_data=load_cached_data)

    # Initialize Model
    model = SkyMotionModel()

    # Initialize Trainer
    trainer = Trainer(model)

    # Train
    trainer.fit(train_loader, val_loader)

    return trainer


def generate_submission(trainer=None, load_cached_data=True):
    set_seed()

    # If trainer is not provided, initialize a dummy one just to load the model for prediction
    if trainer is None:
        model = SkyMotionModel()
        trainer = Trainer(model)

    # Get Test Loader and Metadata
    test_loader, test_meta = get_test_loader(load_cached_data=load_cached_data)

    # Predict residuals (d_lat_m, d_lon_m)
    pred_residuals = trainer.predict(test_loader)

    print("Reconstructing absolute coordinates...")

    # Reconstruct absolute coordinates
    # lat_pred = lat_wls + meters_to_degrees(d_lat_m)
    # lon_pred = lon_wls + meters_to_degrees(d_lon_m)

    wls_lat = test_meta["wls_lat"].values
    wls_lon = test_meta["wls_lon"].values

    d_lat_m = pred_residuals[:, 0]
    d_lon_m = pred_residuals[:, 1]

    # Vectorized conversion
    d_lat_deg, d_lon_deg = meters_to_degrees_diff(d_lat_m, d_lon_m, wls_lat)

    pred_lat = wls_lat + d_lat_deg
    pred_lon = wls_lon + d_lon_deg

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": test_meta["tripId"],
            "UnixTimeMillis": test_meta["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # Save submission
    submission_df.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_OUTPUT_PATH}")
    print(submission_df.head())
