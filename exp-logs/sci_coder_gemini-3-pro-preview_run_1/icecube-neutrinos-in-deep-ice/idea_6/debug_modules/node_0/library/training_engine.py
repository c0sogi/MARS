import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    set_seed,
    angular_dist_score,
    spherical_to_cartesian,
    cartesian_to_spherical,
)
from library.data_processing import IceCubeDataset, collate_fn
from library.model_architecture import DynGTNet


class CosineDirectionLoss(nn.Module):
    """
    Loss function that minimizes 1 - cosine_similarity(pred, target).
    Expects predictions in Cartesian coordinates (x, y, z) and targets in Spherical (azimuth, zenith).
    """

    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, pred_vector, target_angles):
        """
        Args:
            pred_vector: (Batch, 3) predicted vectors (logits, not necessarily normalized).
            target_angles: (Batch, 2) target [azimuth, zenith] in radians.
        """
        # Unpack targets
        azimuth = target_angles[:, 0]
        zenith = target_angles[:, 1]

        # Convert target spherical to cartesian unit vectors
        target_vector = spherical_to_cartesian(azimuth, zenith)

        # Ensure target is on the same device
        if isinstance(target_vector, torch.Tensor):
            target_vector = target_vector.to(pred_vector.device)

        # Compute cosine similarity
        # Note: nn.CosineSimilarity automatically normalizes the inputs
        similarity = self.cos(pred_vector, target_vector)

        # Loss is 1 - mean similarity
        loss = 1.0 - similarity.mean()
        return loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.criterion = CosineDirectionLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # OneCycleLR Scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            epochs=config.MAX_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        self.best_val_score = float("inf")
        self.best_model_path = os.path.join(config.MODEL_DIR, "best_model.pth")

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        count = 0

        start_time = time.time()

        for batch in self.train_loader:
            # Unpack batch
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            self.optimizer.zero_grad()

            # Forward
            preds = self.model(x)

            # Loss
            loss = self.criterion(preds, y)

            # Backward
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item() * x.size(0)
            count += x.size(0)

        avg_loss = running_loss / count
        duration = time.time() - start_time

        print(
            f"Epoch {epoch_idx+1}/{self.config.MAX_EPOCHS} - Train Loss: {avg_loss} - Time: {duration:.2f}s"
        )
        return avg_loss

    def validate(self):
        self.model.eval()
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                # Forward
                pred_vecs = self.model(x)

                # Store results
                preds_list.append(pred_vecs.cpu())
                targets_list.append(y.cpu())

        # Concatenate
        preds_tensor = torch.cat(preds_list, dim=0)
        targets_tensor = torch.cat(targets_list, dim=0)

        # Convert predictions (x, y, z) to (azimuth, zenith)
        pred_x = preds_tensor[:, 0]
        pred_y = preds_tensor[:, 1]
        pred_z = preds_tensor[:, 2]

        pred_az, pred_zen = cartesian_to_spherical(pred_x, pred_y, pred_z)

        # Stack for metric calculation: (N, 2)
        y_pred_angles = torch.stack([pred_az, pred_zen], dim=1).numpy()
        y_true_angles = targets_tensor.numpy()

        # Calculate Metric
        score = angular_dist_score(y_true_angles, y_pred_angles)
        return score

    def fit(self):
        print(f"Starting training on device: {self.device}")
        patience_counter = 0

        for epoch in range(self.config.MAX_EPOCHS):
            _ = self.train_one_epoch(epoch)

            val_score = self.validate()
            print(f"Epoch {epoch+1} Validation Mean Angular Error: {val_score}")

            # Checkpoint
            if val_score < self.best_val_score:
                print(
                    f"Validation score improved from {self.best_val_score} to {val_score}. Saving model..."
                )
                self.best_val_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{self.config.PATIENCE}"
                )

            if patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {self.best_val_score}")

    def predict_and_submit(self, test_loader):
        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        all_event_ids = []
        all_azimuths = []
        all_zeniths = []

        print("Generating predictions on test set...")
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(self.device)
                event_ids = batch["event_ids"]

                # Forward
                pred_vecs = self.model(x)

                # Convert to CPU
                pred_vecs = pred_vecs.cpu()

                # Convert to spherical
                pred_x = pred_vecs[:, 0]
                pred_y = pred_vecs[:, 1]
                pred_z = pred_vecs[:, 2]

                az, zen = cartesian_to_spherical(pred_x, pred_y, pred_z)

                all_event_ids.extend(event_ids)
                all_azimuths.extend(az.numpy())
                all_zeniths.extend(zen.numpy())

        # Create DataFrame
        df_submission = pd.DataFrame(
            {"event_id": all_event_ids, "azimuth": all_azimuths, "zenith": all_zeniths}
        )

        # Sort by event_id just in case, though usually not strictly required if all present
        df_submission = df_submission.sort_values("event_id")

        # Save
        save_path = os.path.join(self.config.SUBMISSION_DIR, "submission.csv")
        df_submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        return df_submission


def run_training():
    """
    Main entry point to run the training pipeline.
    """
    set_seed(Config.SEED)
    Config.setup()

    # 1. Prepare Datasets
    print("Initializing Datasets...")
    train_dataset = IceCubeDataset(mode="train")
    val_dataset = IceCubeDataset(mode="val")

    # 2. Prepare Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Model
    print("Initializing DynGTNet...")
    model = DynGTNet()

    # 4. Trainer
    trainer = Trainer(model, train_loader, val_loader, Config)

    # 5. Train
    trainer.fit()

    # 6. Inference
    print("Initializing Test Dataset...")
    test_dataset = IceCubeDataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    trainer.predict_and_submit(test_loader)
