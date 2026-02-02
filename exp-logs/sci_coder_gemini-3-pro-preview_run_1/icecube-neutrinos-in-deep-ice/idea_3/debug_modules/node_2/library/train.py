import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config, set_seed
from library.data import get_dataloaders
from library.model import HybridRecurrentDenseNet
from library.utils import cartesian_to_spherical


class CosineSimilarityLoss(nn.Module):
    """
    Computes the Cosine Similarity Loss between predicted 3D vectors and target spherical angles.
    Loss = 1 - mean(cosine_similarity(pred, target))
    """

    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, pred_vector, target_angles):
        """
        Args:
            pred_vector: (Batch, 3) Cartesian coordinates (x, y, z).
            target_angles: (Batch, 2) Spherical coordinates (azimuth, zenith).
        Returns:
            Scalar loss.
        """
        # Unpack targets
        azimuth = target_angles[:, 0]
        zenith = target_angles[:, 1]

        # Convert target spherical angles to Cartesian unit vectors
        # x = cos(az) * sin(zen)
        # y = sin(az) * sin(zen)
        # z = cos(zen)
        sin_zen = torch.sin(zenith)
        target_x = torch.cos(azimuth) * sin_zen
        target_y = torch.sin(azimuth) * sin_zen
        target_z = torch.cos(zenith)

        target_vector = torch.stack([target_x, target_y, target_z], dim=1)

        # Normalize predicted vectors to unit length
        pred_norm = F.normalize(pred_vector, p=2, dim=1)

        # Compute cosine similarity: dot product of unit vectors
        # shape: (Batch,)
        cosine_sim = torch.sum(pred_norm * target_vector, dim=1)

        # Loss = 1 - mean similarity
        # Similarity is in [-1, 1], so Loss is in [0, 2]
        loss = 1.0 - torch.mean(cosine_sim)

        return loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = CosineSimilarityLoss()
        self.best_val_mae = float("inf")

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (seq, features, targets) in enumerate(self.train_loader):
            seq = seq.to(self.device, non_blocking=True)
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            outputs = self.model(seq, features)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        total_mae = 0.0
        num_samples = 0

        with torch.no_grad():
            for seq, features, targets in self.val_loader:
                seq = seq.to(self.device, non_blocking=True)
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(seq, features)

                # Compute Loss
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * seq.size(0)

                # Compute MAE
                # 1. Convert targets to vectors
                azimuth = targets[:, 0]
                zenith = targets[:, 1]
                sin_zen = torch.sin(zenith)
                tx = torch.cos(azimuth) * sin_zen
                ty = torch.sin(azimuth) * sin_zen
                tz = torch.cos(zenith)
                t_vec = torch.stack([tx, ty, tz], dim=1)

                # 2. Normalize predictions
                p_vec = F.normalize(outputs, p=2, dim=1)

                # 3. Dot product
                dot_prod = torch.sum(p_vec * t_vec, dim=1)

                # 4. Clamp for numerical stability
                dot_prod = torch.clamp(dot_prod, -1.0 + 1e-7, 1.0 - 1e-7)

                # 5. Angular error
                angles = torch.acos(dot_prod)
                total_mae += torch.sum(angles).item()
                num_samples += seq.size(0)

        avg_loss = running_loss / num_samples
        avg_mae = total_mae / num_samples
        return avg_loss, avg_mae

    def fit(self, epochs, patience, save_path):
        print(f"Starting training for {epochs} epochs with patience {patience}...")

        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_mae = self.validate()

            duration = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val MAE: {val_mae}"
            )

            # Early Stopping and Model Saving
            if val_mae < self.best_val_mae:
                print(
                    f"Validation MAE improved from {self.best_val_mae} to {val_mae}. Saving model..."
                )
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation MAE: {self.best_val_mae}")


def run_training_and_inference(load_cached_data=True):
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure directories exist
    Config.setup()

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    model = HybridRecurrentDenseNet().to(device)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate total steps for OneCycleLR
    steps_per_epoch = len(train_loader)
    total_steps = Config.EPOCHS * steps_per_epoch

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 5. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    trainer.fit(
        epochs=Config.EPOCHS, patience=Config.PATIENCE, save_path=Config.MODEL_SAVE_PATH
    )

    # 6. Inference
    print("Starting Inference on Test Set...")

    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model weights.")
    else:
        print("Warning: Model checkpoint not found. Using current weights.")

    model.eval()

    all_event_ids = []
    all_azimuths = []
    all_zeniths = []

    with torch.no_grad():
        for seq, features, event_ids in test_loader:
            seq = seq.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)

            # Predict vector (Batch, 3)
            outputs = model(seq, features)

            # Move to CPU numpy
            vectors = outputs.cpu().numpy()

            # Convert to Spherical
            # vectors[:, 0] -> x, vectors[:, 1] -> y, vectors[:, 2] -> z
            az, zen = cartesian_to_spherical(
                vectors[:, 0], vectors[:, 1], vectors[:, 2]
            )

            all_event_ids.extend(event_ids.numpy())
            all_azimuths.extend(az)
            all_zeniths.extend(zen)

    # 7. Save Submission
    print("Saving submission...")
    submission_df = pd.DataFrame(
        {"event_id": all_event_ids, "azimuth": all_azimuths, "zenith": all_zeniths}
    )

    # Ensure event_id is int
    submission_df["event_id"] = submission_df["event_id"].astype(int)

    # Sort by event_id to match sample submission style (optional but good practice)
    submission_df = submission_df.sort_values("event_id")

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
