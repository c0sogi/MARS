import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model import ResFunnelGLU


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Manages the training, validation, and inference processes for the ResFunnelGLU model.
    """

    def __init__(self, model, device=Config.DEVICE):
        """
        Args:
            model (nn.Module): The neural network model to train.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.device = device
        self.save_path = Config.MODEL_SAVE_PATH

        # Optimization setup
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def train_epoch(self, loader):
        """
        Runs one epoch of training.

        Args:
            loader (DataLoader): Training data loader.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        dataset_size = len(loader.dataset)

        for batch in loader:
            cont = batch["cont"].to(self.device)
            cat = batch["cat"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(cont, cat)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * cont.size(0)

        return total_loss / dataset_size

    def validate(self, loader):
        """
        Runs validation on the provided loader.

        Args:
            loader (DataLoader): Validation data loader.

        Returns:
            tuple: (average_loss, auc_score)
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []
        dataset_size = len(loader.dataset)

        with torch.no_grad():
            for batch in loader:
                cont = batch["cont"].to(self.device)
                cat = batch["cat"].to(self.device)
                target = batch["target"].to(self.device)

                output = self.model(cont, cat)
                loss = self.criterion(output, target)

                total_loss += loss.item() * cont.size(0)
                all_targets.append(target.cpu().numpy())
                all_preds.append(output.cpu().numpy())

        avg_loss = total_loss / dataset_size

        # Concatenate all batches
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)

        # Calculate AUC
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            # Handle edge cases where only one class is present in the batch
            auc = 0.5

        return avg_loss, auc

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.MAX_EPOCHS,
        patience=Config.PATIENCE,
    ):
        """
        Executes the full training loop with early stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.

        Returns:
            float: The best Validation AUC achieved.
        """
        best_auc = 0.0
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Checkpointing based on AUC
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"New best model saved with AUC: {best_auc}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Best Val AUC: {best_auc}")
        return best_auc

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set and saves them to a CSV file.

        Args:
            test_loader (DataLoader): Test data loader.
        """
        print("Generating submission...")

        # Attempt to load the best model weights
        if os.path.exists(self.save_path):
            print(f"Loading best model from {self.save_path}...")
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=self.device)
            )
        else:
            print("Warning: Best model file not found. Using current model weights.")

        self.model.eval()
        all_preds = []

        # Inference loop
        with torch.no_grad():
            for batch in test_loader:
                cont = batch["cont"].to(self.device)
                cat = batch["cat"].to(self.device)

                output = self.model(cont, cat)
                all_preds.append(output.cpu().numpy())

        # Flatten predictions
        all_preds = np.concatenate(all_preds).flatten()

        # Load Test Metadata to get IDs
        if not os.path.exists(Config.TEST_META_PATH):
            raise FileNotFoundError(
                f"Test metadata not found at {Config.TEST_META_PATH}"
            )

        test_meta = pd.read_csv(Config.TEST_META_PATH)
        test_ids = test_meta["id"]

        if len(all_preds) != len(test_ids):
            print(
                f"Warning: Number of predictions ({len(all_preds)}) does not match number of test IDs ({len(test_ids)})."
            )

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "target": all_preds})

        # Save to disk
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
