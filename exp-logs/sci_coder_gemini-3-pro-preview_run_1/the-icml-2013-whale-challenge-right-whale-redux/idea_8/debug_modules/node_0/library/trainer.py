import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion, compute_auc
from library.model import CoordinateAttentionCRNN
from library.dataset import get_dataloaders


class Trainer:
    """
    Trainer class for the Coordinate Attention CRNN model.
    Handles training, validation, early stopping, and inference.
    """

    def __init__(self):
        # Reproducibility
        set_seed(Config.SEED)

        self.device = Config.DEVICE

        # Initialize Model
        self.model = CoordinateAttentionCRNN().to(self.device)

        # Loss Function with Class Weighting
        # We wrap pos_weight in a tensor and move to device
        pos_weight = torch.tensor([Config.POS_WEIGHT]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (Maximize AUC)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True,
        )

    def train_one_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training with Mixup augmentation.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)

            # Apply Mixup
            mixed_data, target_a, target_b, lam = mixup_data(
                data, target, alpha=Config.MIXUP_ALPHA, device=self.device
            )

            # Forward Pass
            self.optimizer.zero_grad()
            output = self.model(mixed_data)

            # Calculate Mixup Loss
            # Output shape is (Batch, 1), target is (Batch,)
            # Squeeze output to match target shape
            loss = mixup_criterion(
                self.criterion, output.squeeze(1), target_a, target_b, lam
            )

            # Backward Pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation on the validation set (no augmentation).
        Returns average loss and ROC AUC.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)

                # Forward Pass
                output = self.model(data)
                logits = output.squeeze(1)

                # Calculate Loss (Standard BCE)
                loss = self.criterion(logits, target)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Store predictions for AUC
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                all_targets.append(target.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        epoch_loss = running_loss / dataset_size

        # Concatenate all batches
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        # Compute AUC
        epoch_auc = compute_auc(all_targets, all_preds)

        return epoch_loss, epoch_auc

    def run(self):
        """
        Main execution method:
        1. Get DataLoaders
        2. Train loop with Early Stopping
        3. Save best model
        4. Generate submission
        """
        print(f"Starting training on device: {self.device}")

        # 1. Get DataLoaders
        train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

        best_auc = 0.0
        patience_counter = 0

        # 2. Training Loop
        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_auc = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step(val_auc)

            duration = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc}"
            )

            # Early Stopping & Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"New best model saved with AUC: {best_auc}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation AUC: {best_auc}")

        # 3. Prediction / Submission
        self.predict(test_loader)

    def predict(self, test_loader):
        """
        Loads the best model and generates predictions for the test set.
        Saves the result to Config.SUBMISSION_PATH.
        """
        print("Starting inference on test set...")

        # Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            print("Error: Best model file not found.")
            return

        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        results = []

        with torch.no_grad():
            for data, ids in test_loader:
                data = data.to(self.device)

                # Forward Pass
                output = self.model(data)
                logits = output.squeeze(1)
                probs = torch.sigmoid(logits)

                probs_np = probs.cpu().numpy()

                # Collect results
                for clip_id, prob in zip(ids, probs_np):
                    results.append({"clip": clip_id, "probability": prob})

        # Create DataFrame
        df_submission = pd.DataFrame(results)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_submission.shape}")
        print("Head of submission:")
        print(df_submission.head())
