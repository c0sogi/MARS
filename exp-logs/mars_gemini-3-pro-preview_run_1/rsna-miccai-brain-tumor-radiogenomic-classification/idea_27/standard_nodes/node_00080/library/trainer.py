import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    SUBMISSION_PATH,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    DEBUG_DATA_LIMIT,
    SEED,
)
from library.utils import seed_everything, AverageMeter, print_header, print_metric
from library.dataset import get_dataloader
from library.model import SIRVEfficientNet


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the SIRV model.
    """

    def __init__(self):
        self.device = torch.device(DEVICE)
        self.model_save_path = os.path.join(WORKING_DIR, "best_model.pth")

        # Initialize Model
        self.model = SIRVEfficientNet(pretrained=True)
        self.model.to(self.device)

        # Optimization Components
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
        )

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)  # (B, 1)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, val_loader):
        """
        Runs validation and computes ROC AUC.
        """
        self.model.eval()
        losses = AverageMeter()
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(logits)

                losses.update(loss.item(), images.size(0))
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        # Handle edge case if only one class is present in batch/subset
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc = 0.5

        return losses.avg, auc

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        seed_everything(SEED)
        print_header("Starting Training Pipeline")

        # 1. Load Metadata
        df_train = pd.read_csv(TRAIN_METADATA_PATH)
        df_val = pd.read_csv(VAL_METADATA_PATH)

        # Debugging limit
        if DEBUG_DATA_LIMIT is not None:
            print(f"DEBUG: Limiting training data to {DEBUG_DATA_LIMIT} samples.")
            df_train = df_train.head(DEBUG_DATA_LIMIT)
            df_val = df_val.head(DEBUG_DATA_LIMIT)

        # 2. Create DataLoaders
        train_loader = get_dataloader(df_train, phase="train")
        val_loader = get_dataloader(df_val, phase="valid")

        best_auc = 0.0
        patience = 5
        patience_counter = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_auc = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()

            # Logging
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}/{NUM_EPOCHS} [{elapsed:.0f}s] | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )
            print_metric("Validation AUC", val_auc)

            # Checkpoint & Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"New best model saved with AUC: {best_auc}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print_header("Training Complete")
        print(f"Best Validation AUC: {best_auc}")

    def predict(self):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        print_header("Starting Inference")

        # 1. Load Test Metadata
        df_test = pd.read_csv(TEST_METADATA_PATH)

        if DEBUG_DATA_LIMIT is not None:
            print(f"DEBUG: Limiting test data to {DEBUG_DATA_LIMIT} samples.")
            df_test = df_test.head(DEBUG_DATA_LIMIT)

        test_loader = get_dataloader(df_test, phase="test", shuffle=False)

        # 2. Load Best Model
        if not os.path.exists(self.model_save_path):
            print("Error: No trained model found to load.")
            return

        self.model.load_state_dict(
            torch.load(self.model_save_path, map_location=self.device)
        )
        self.model.eval()

        predictions = []
        ids = []

        # 3. Inference Loop
        with torch.no_grad():
            for images, subject_ids in test_loader:
                images = images.to(self.device)

                logits = self.model(images)
                probs = torch.sigmoid(logits)

                predictions.extend(probs.cpu().numpy().flatten())
                ids.extend(subject_ids)

        # 4. Create Submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

        # Ensure BraTS21ID is handled correctly (as strings or ints matching sample)
        # The sample submission usually expects IDs like 00001 or 1.
        # Based on dataset info, IDs are ints.
        submission_df["BraTS21ID"] = submission_df["BraTS21ID"].astype(int)

        # Save
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(submission_df.head())


def main():
    trainer = Trainer()
    trainer.fit()
    trainer.predict()


# Note: The 'if __name__ == "__main__":' block is excluded as per instructions.
# The main() function is provided for clarity on how to run the pipeline.
