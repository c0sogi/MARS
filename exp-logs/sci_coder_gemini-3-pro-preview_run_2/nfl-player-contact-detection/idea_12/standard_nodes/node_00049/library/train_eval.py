import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, compute_mcc, FocalLoss
from library.data_processing import DataProcessor
from library.model import ECPIRN


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the EC-PIRN model.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.model = ECPIRN().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.criterion = FocalLoss(
            alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
        )
        self.best_threshold = 0.5

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)  # (Batch, 1)

            self.optimizer.zero_grad()

            # Forward pass (returns logits)
            logits = self.model(inputs)

            loss = self.criterion(logits, targets)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation and returns loss, true labels, and raw logits.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_logits = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * inputs.size(0)

                all_targets.append(targets.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)
        all_targets = np.concatenate(all_targets).flatten()
        all_logits = np.concatenate(all_logits).flatten()

        return val_loss, all_targets, all_logits

    def optimize_threshold(self, targets, logits):
        """
        Performs a grid search to find the threshold that maximizes MCC.
        """
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_mcc = -1.0
        best_th = 0.5

        for th in thresholds:
            preds = (probs >= th).astype(int)
            score = compute_mcc(targets, preds)
            if score > best_mcc:
                best_mcc = score
                best_th = th

        return best_th, best_mcc

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    ):
        """
        Full training loop with Early Stopping based on Validation MCC.
        """
        best_val_mcc = -1.0
        patience_counter = 0
        best_model_state = None

        print(f"Starting training on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_targets, val_logits = self.validate(val_loader)

            # Find best threshold for this epoch to evaluate performance
            curr_best_th, curr_val_mcc = self.optimize_threshold(
                val_targets, val_logits
            )

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MCC: {curr_val_mcc:.10f} | "
                f"Best Th: {curr_best_th:.2f}"
            )

            # Early Stopping Logic
            if curr_val_mcc > best_val_mcc:
                best_val_mcc = curr_val_mcc
                self.best_threshold = curr_best_th
                best_model_state = self.model.state_dict()
                patience_counter = 0
                # Save best model checkpoint
                torch.save(
                    best_model_state, os.path.join(Config.WORKING_DIR, "best_model.pth")
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(
                f"Loaded best model with Val MCC: {best_val_mcc:.10f} at Threshold: {self.best_threshold:.2f}"
            )

    def predict(self, test_loader):
        """
        Generates binary predictions for the test set using the optimized threshold.
        """
        self.model.eval()
        all_logits = []

        with torch.no_grad():
            for inputs in test_loader:
                # inputs is a list/tuple from TensorDataset, taking the first element
                x = inputs[0].to(self.device)
                logits = self.model(x)
                all_logits.append(logits.cpu().numpy())

        all_logits = np.concatenate(all_logits).flatten()
        probs = 1 / (1 + np.exp(-all_logits))
        predictions = (probs >= self.best_threshold).astype(int)

        return predictions


def run_pipeline(load_cached_data=True):
    """
    Orchestrates the data loading, training, and submission generation.
    """
    seed_everything(Config.SEED)

    # 1. Data Processing
    processor = DataProcessor()

    # Load Train/Val
    X_train, y_train, X_val, y_val = processor.get_train_val_data(
        load_cached_data=load_cached_data
    )

    # Convert to Tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Training
    trainer = Trainer()
    trainer.fit(train_loader, val_loader)

    # 3. Inference
    X_test, df_test_meta = processor.get_test_data(load_cached_data=load_cached_data)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    predictions = trainer.predict(test_loader)

    # 4. Submission
    df_submission = pd.DataFrame(
        {"contact_id": df_test_meta["contact_id"], "contact": predictions}
    )

    save_path = Config.SUBMISSION_PATH
    df_submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(df_submission.head())
