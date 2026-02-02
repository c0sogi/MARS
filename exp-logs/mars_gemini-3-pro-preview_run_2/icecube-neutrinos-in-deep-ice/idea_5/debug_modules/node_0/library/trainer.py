import os
import time
import torch
import numpy as np
import torch.nn as nn
from library.config import Config
from library.model import DualStreamNetwork, CosineSimilarityLoss
from library.utils import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    angular_dist_score,
    get_cosine_schedule_with_warmup,
)


class IceCubeTrainer:
    """
    Trainer class for the Dual-Stream Geometric-Temporal Network.
    Encapsulates training, validation, and checkpointing logic.
    """

    def __init__(self, config, train_loader, val_loader):
        """
        Args:
            config: Configuration object with hyperparameters and paths.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.DEVICE)

        # Initialize Model
        self.model = DualStreamNetwork().to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Function
        self.criterion = CosineSimilarityLoss()

        # Scheduler
        # Calculate total steps for the scheduler
        num_training_steps = len(train_loader) * config.NUM_EPOCHS
        num_warmup_steps = len(train_loader) * config.WARMUP_EPOCHS

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Tracking variables
        self.best_val_score = float("inf")
        self.best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    def train_epoch(self):
        """
        Runs one epoch of training.
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        train_losses = []

        for batch_idx, (seq_x, geom_x, target_angles, _) in enumerate(
            self.train_loader
        ):
            seq_x = seq_x.to(self.device)
            geom_x = geom_x.to(self.device)

            # Prepare targets: (az, zen) -> (x, y, z) vector
            # We need cartesian vectors for the CosineSimilarityLoss
            azimuth = target_angles[:, 0].to(self.device)
            zenith = target_angles[:, 1].to(self.device)

            target_vecs_x, target_vecs_y, target_vecs_z = spherical_to_cartesian(
                azimuth, zenith
            )
            target_vecs = torch.stack(
                [target_vecs_x, target_vecs_y, target_vecs_z], dim=1
            )

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(seq_x, geom_x)

            # Compute loss
            loss = self.criterion(preds, target_vecs)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_losses.append(loss.item())

        return np.mean(train_losses)

    def validate(self):
        """
        Runs validation on the validation set.
        Returns:
            float: Mean Angular Error (MAE) on the validation set.
        """
        self.model.eval()

        val_preds_az = []
        val_preds_zen = []
        val_true_az = []
        val_true_zen = []

        with torch.no_grad():
            for seq_x, geom_x, target_angles, _ in self.val_loader:
                seq_x = seq_x.to(self.device)
                geom_x = geom_x.to(self.device)

                # Forward pass
                preds_vec = self.model(seq_x, geom_x)

                # Convert predictions (vectors) back to spherical for metric calculation
                p_x = preds_vec[:, 0]
                p_y = preds_vec[:, 1]
                p_z = preds_vec[:, 2]
                pred_az, pred_zen = cartesian_to_spherical(p_x, p_y, p_z)

                # Store predictions and targets (move to CPU numpy)
                val_preds_az.append(pred_az.cpu().numpy())
                val_preds_zen.append(pred_zen.cpu().numpy())
                val_true_az.append(target_angles[:, 0].numpy())
                val_true_zen.append(target_angles[:, 1].numpy())

        # Concatenate all batches
        val_preds_az = np.concatenate(val_preds_az)
        val_preds_zen = np.concatenate(val_preds_zen)
        val_true_az = np.concatenate(val_true_az)
        val_true_zen = np.concatenate(val_true_zen)

        # Calculate Metric (Mean Angular Error)
        score = angular_dist_score(
            val_true_az, val_true_zen, val_preds_az, val_preds_zen
        )
        return score

    def fit(self):
        """
        Main training loop with early stopping.
        Returns:
            str: Path to the best saved model.
        """
        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(self.config.NUM_EPOCHS):
            start_time = time.time()

            # Train
            avg_train_loss = self.train_epoch()

            # Validate
            val_score = self.validate()

            elapsed = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} | "
                f"Train Loss: {avg_train_loss} | "
                f"Val MAE: {val_score} | "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpointing & Early Stopping
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with MAE: {self.best_val_score}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {self.best_val_score}")
        return self.best_model_path
