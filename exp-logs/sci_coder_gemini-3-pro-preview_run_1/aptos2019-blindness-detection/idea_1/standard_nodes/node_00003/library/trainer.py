import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from library.config import Config
from library.utils import seed_everything, ordinal_decode, compute_qwk
from library.data import get_dataloaders
from library.model import OrdinalMobileNetV3


class Trainer:
    def __init__(self, model, device=Config.DEVICE):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The model to train.
            device (torch.device): Device to run training on.
        """
        self.model = model.to(device)
        self.device = device

        # Loss function: Binary Cross Entropy with Logits
        # We use this because the model outputs logits for 4 independent binary tasks (P(y>k))
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        self.best_qwk = -float("inf")
        self.best_loss = float("inf")

    def train_one_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Returns loss and QWK score.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                batch_size = images.size(0)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                # Decode predictions (sum probabilities and round)
                preds = ordinal_decode(probs)

                # Decode targets (sum binary vector to get original integer class)
                # target vector [1, 1, 0, 0] sums to 2.
                true_labels = torch.sum(targets, dim=1).cpu().numpy().astype(int)

                all_preds.extend(preds)
                all_targets.extend(true_labels)

        epoch_loss = running_loss / dataset_size
        qwk = compute_qwk(all_targets, all_preds)

        return epoch_loss, qwk

    def fit(
        self, train_loader, val_loader, epochs=Config.EPOCHS, patience=Config.PATIENCE
    ):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs on {self.device}...")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_qwk = self.evaluate(val_loader)

            # Step the scheduler
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val QWK: {val_qwk}"
            )

            # Early Stopping Logic based on QWK (Maximize)
            if val_qwk > self.best_qwk:
                self.best_qwk = val_qwk
                patience_counter = 0
                self.save_checkpoint(Config.MODEL_SAVE_PATH)
                print(f"New best QWK! Model saved to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val QWK: {self.best_qwk}")

    def save_checkpoint(self, path):
        """Saves the model state dict."""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        """Loads the model state dict."""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")
        else:
            print(f"Warning: No checkpoint found at {path}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Returns:
            pd.DataFrame: DataFrame with 'id_code' and 'diagnosis'.
        """
        self.model.eval()
        predictions = []

        # We need to map predictions back to id_codes.
        # The test_loader iterates sequentially.
        # We load the test metadata to get the IDs in order.
        df_test = pd.read_csv(Config.TEST_META_PATH)
        id_codes = df_test["id_code"].values

        idx_counter = 0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)

                logits = self.model(images)
                probs = torch.sigmoid(logits)
                preds = ordinal_decode(probs)

                # Append to list
                # preds is a numpy array of integers
                predictions.extend(preds)

        # Ensure lengths match
        if len(predictions) != len(id_codes):
            print(
                f"Warning: Prediction count ({len(predictions)}) matches ID count ({len(id_codes)})?"
            )
            # Truncate or pad if necessary, though logic dictates they should match
            min_len = min(len(predictions), len(id_codes))
            predictions = predictions[:min_len]
            id_codes = id_codes[:min_len]

        submission_df = pd.DataFrame({"id_code": id_codes, "diagnosis": predictions})

        return submission_df


def run_training_pipeline(debug=False):
    """
    Orchestrates the entire training and submission pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Data
    sample_size = Config.DEBUG_SAMPLE_SIZE if debug else None
    dataloaders = get_dataloaders(debug_sample_size=sample_size)

    # 3. Model
    model = OrdinalMobileNetV3(pretrained=Config.PRETRAINED)

    # 4. Trainer
    trainer = Trainer(model)

    # 5. Train
    trainer.fit(dataloaders["train"], dataloaders["val"])

    # 6. Predict (Load best model first)
    trainer.load_checkpoint(Config.MODEL_SAVE_PATH)
    submission_df = trainer.predict(dataloaders["test"])

    # 7. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
