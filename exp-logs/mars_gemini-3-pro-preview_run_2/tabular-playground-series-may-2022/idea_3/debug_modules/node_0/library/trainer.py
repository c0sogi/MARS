import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.dataset import get_datasets
from library.model import ResMLP


def set_seed(seed=Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Manages the training, evaluation, and inference of the ResMLP model.
    """

    def __init__(self):
        # Ensure reproducibility
        set_seed()

        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = ResMLP().to(self.device)

        # Initialize Optimizer (AdamW with weight decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function (Binary Cross Entropy with Logits)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_one_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            # Move data to device
            continuous = batch["continuous"].to(self.device)
            categorical = batch["categorical"].to(self.device)
            targets = batch["target"].to(self.device).unsqueeze(1)  # (B, 1)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(continuous, categorical)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set using ROC AUC.
        """
        self.model.eval()
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                continuous = batch["continuous"].to(self.device)
                categorical = batch["categorical"].to(self.device)
                targets = batch["target"].to(self.device)

                # Forward pass
                outputs = self.model(continuous, categorical)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs).squeeze(1)

                all_targets.append(targets.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        # Concatenate all batches
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        # Compute AUC
        auc = roc_auc_score(all_targets, all_preds)
        return auc

    def fit(self, epochs=Config.EPOCHS, patience=5, load_cached_data=True):
        """
        Main training loop with Early Stopping.
        """
        print(f"Loading datasets (Cached: {load_cached_data})...")
        train_ds, val_ds, _ = get_datasets(load_cached_data=load_cached_data)

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Starting training on {self.device} for {epochs} epochs...")

        best_auc = -float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_auc = self.evaluate(val_loader)

            # Print metrics (Full precision for AUC)
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered. No improvement for {patience} epochs."
                    )
                    break

        print(f"Training complete. Best Val AUC: {best_auc}")

    def generate_submission(self, load_cached_data=True):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Generating submission...")

        # Load best model state
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            raise FileNotFoundError(
                f"Model file not found at {Config.MODEL_SAVE_PATH}. Run fit() first."
            )

        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        # Load Test Data
        _, _, test_ds = get_datasets(load_cached_data=load_cached_data)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                continuous = batch["continuous"].to(self.device)
                categorical = batch["categorical"].to(self.device)

                outputs = self.model(continuous, categorical)
                probs = torch.sigmoid(outputs).squeeze(1)

                all_preds.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        # Load Test Metadata to get IDs
        test_meta = pd.read_csv(Config.TEST_METADATA)

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_meta["id"], "target": all_preds})

        # Save to CSV
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
