import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.model import RSNAModel
from library.data import get_loaders
from library.utils import weighted_loss


class RSNAEngine:
    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.model = RSNAModel(pretrained=True).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        self.best_loss = float("inf")
        self.current_epoch = 0

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        self.optimizer.zero_grad()

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            # Mixed Precision Forward Pass
            with autocast():
                logits = self.model(images)
                loss = weighted_loss(logits, targets)
                # Normalize loss for gradient accumulation
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

            # Backward Pass
            self.scaler.scale(loss).backward()

            # Gradient Accumulation Step
            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Track Loss (multiply back to get actual batch loss)
            running_loss += (
                loss.item() * Config.GRADIENT_ACCUMULATION_STEPS * batch_size
            )
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                batch_size = images.size(0)

                with autocast():
                    logits = self.model(images)
                    loss = weighted_loss(logits, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def fit(self, train_loader, val_loader):
        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            self.current_epoch = epoch + 1

            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()

            # Print Metrics (Full Precision)
            print(
                f"Epoch {self.current_epoch}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Checkpoint & Early Stopping
            if val_loss < (self.best_loss - Config.EARLY_STOPPING_MIN_DELTA):
                self.best_loss = val_loss
                patience_counter = 0

                # Save Best Model
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")
            else:
                patience_counter += 1
                print(
                    f"EarlyStopping counter: {patience_counter} out of {Config.EARLY_STOPPING_PATIENCE}"
                )

                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

    def predict_and_submit(self, test_loader):
        print("Generating submission...")

        # Load Best Model
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model for inference.")
        else:
            print("Warning: Best model not found, using current model state.")

        self.model.eval()

        results = []
        # Column mapping based on Config: [C1, C2, C3, C4, C5, C6, C7, Patient_Overall]
        target_suffixes = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

        with torch.no_grad():
            for images, study_uids in test_loader:
                images = images.to(self.device, non_blocking=True)

                with autocast():
                    logits = self.model(images)
                    probs = torch.sigmoid(logits)

                probs = probs.cpu().numpy()

                # Unpack batch
                for i, uid in enumerate(study_uids):
                    patient_probs = probs[i]

                    for idx, suffix in enumerate(target_suffixes):
                        row_id = f"{uid}_{suffix}"
                        prob = patient_probs[idx]
                        results.append({"row_id": row_id, "fractured": prob})

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # Save to CSV
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        return submission_df


def run():
    """
    Main entry point to execute the training and submission pipeline.
    """
    # 1. Get DataLoaders
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 2. Initialize Engine
    engine = RSNAEngine()

    # 3. Train
    engine.fit(train_loader, val_loader)

    # 4. Generate Submission
    engine.predict_and_submit(test_loader)
