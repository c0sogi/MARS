import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    MAX_EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SUBMISSION_PATH,
    TEST_METADATA_PATH,
    SEED,
)
from library.model import AsymmetricGroupedEfficientNet
from library.data_loader import get_dataloaders
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(SEED)


class CircuitBreaker:
    """
    Monitors data integrity. Aborts if corruption exceeds threshold.
    """

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def check(self, dataset, name="Dataset"):
        """
        Checks the dataset for zero-filled tensors which indicate loading failure.
        Args:
            dataset: BraTSDataset instance containing .images (numpy array)
            name: Name of the dataset for logging
        """
        if not hasattr(dataset, "images"):
            print(f"{name}: Cannot check integrity (no .images attribute). Skipping.")
            return

        images = dataset.images
        total = len(images)
        if total == 0:
            return

        # Check for completely empty volumes (all zeros)
        # Reshape to (N, -1) and sum across pixels/channels
        sums = np.sum(images.reshape(total, -1), axis=1)
        corrupt_count = np.sum(sums == 0)

        corruption_rate = corrupt_count / total

        print(
            f"{name} Integrity Check: {corrupt_count}/{total} corrupt ({corruption_rate:.4f})"
        )

        if corruption_rate > self.threshold:
            raise RuntimeError(
                f"Circuit Breaker Triggered! Corruption rate {corruption_rate:.4f} "
                f"exceeds threshold {self.threshold} for {name}."
            )


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_val_auc = -1.0
        self.patience_counter = 0

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1)

            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)

            # Store for AUC calculation
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(target.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader.dataset)

        # Handle case where single class is present in batch (though unlikely with shuffle)
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

        return epoch_loss, epoch_auc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1)

                logits = self.model(data)
                loss = self.criterion(logits, target)

                running_loss += loss.item() * data.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(target.cpu().numpy())

        val_loss = running_loss / len(self.val_loader.dataset)

        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            val_auc = 0.5

        return val_loss, val_auc

    def fit(self):
        print(f"Starting training on {DEVICE}...")

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_auc = self.train_epoch(epoch)
            val_loss, val_auc = self.validate()

            print(
                f"Epoch {epoch}: Train Loss: {train_loss}, Train AUC: {train_auc}, Val Loss: {val_loss}, Val AUC: {val_auc}"
            )

            # Checkpoint & Early Stopping
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
                print(f"New best model saved with AUC: {val_auc}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best Validation AUC: {self.best_val_auc}")

    def predict_tta(self, test_loader):
        """
        Performs inference using Test-Time Augmentation (TTA).
        Strategies: Original, Horizontal Flip, Vertical Flip.
        """
        print("Loading best model for inference...")
        self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(DEVICE)

                # 1. Original
                logits_orig = self.model(data)
                prob_orig = torch.sigmoid(logits_orig)

                # 2. Horizontal Flip (Flip width dim: last dim)
                data_h = torch.flip(data, dims=[3])
                logits_h = self.model(data_h)
                prob_h = torch.sigmoid(logits_h)

                # 3. Vertical Flip (Flip height dim: 2nd to last dim)
                data_v = torch.flip(data, dims=[2])
                logits_v = self.model(data_v)
                prob_v = torch.sigmoid(logits_v)

                # Average probabilities
                prob_avg = (prob_orig + prob_h + prob_v) / 3.0
                predictions.extend(prob_avg.cpu().numpy().flatten())

        return predictions


def run():
    # 1. Get DataLoaders
    # This will trigger data processing/caching if needed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Circuit Breaker Check
    # We check the underlying datasets for corruption (zero-tensors)
    cb = CircuitBreaker(threshold=0.01)
    cb.check(train_loader.dataset, "Train Set")
    cb.check(val_loader.dataset, "Validation Set")

    # 3. Initialize Model
    model = AsymmetricGroupedEfficientNet()

    # 4. Train
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit()

    # 5. Generate Submission
    print("Generating submission with TTA...")
    preds = trainer.predict_tta(test_loader)

    # Load test metadata to get IDs
    test_df = pd.read_csv(TEST_METADATA_PATH)

    # Ensure lengths match
    if len(preds) != len(test_df):
        print(
            f"Warning: Prediction count {len(preds)} does not match Test ID count {len(test_df)}."
        )
        # Truncate or pad if necessary, though this shouldn't happen with correct loaders
        if len(preds) > len(test_df):
            preds = preds[: len(test_df)]
        else:
            preds = preds + [0.5] * (len(test_df) - len(preds))

    submission = pd.DataFrame({"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": preds})

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    # This block is for local testing of this file only
    run()
