import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys

from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.data_loader import get_dataloaders
from library.model import DualBranchDAN


class Trainer:
    """
    Trainer class to handle training, validation, and prediction for the DualBranchDAN model.
    """

    def __init__(self):
        """
        Initializes the Trainer with model, optimizer, criterion, and device.
        """
        # Set seed for reproducibility
        seed_everything(Config.SEED)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Model
        self.model = DualBranchDAN(
            vocab_size=Config.VOCAB_SIZE,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
            num_targets=len(Config.TARGET_COLS),
        )
        self.model.to(self.device)

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.BCELoss()

        # Placeholders for dataloaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.tokenizer = None

    def train_one_epoch(self, epoch_index):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (q_seq, a_seq, targets) in enumerate(self.train_loader):
            q_seq = q_seq.to(self.device)
            a_seq = a_seq.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(q_seq, a_seq)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns average loss and Spearman correlation score.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for q_seq, a_seq, targets in self.val_loader:
                q_seq = q_seq.to(self.device)
                a_seq = a_seq.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(q_seq, a_seq)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        avg_loss = running_loss / len(self.val_loader)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute Spearman Metric
        spearman_score = compute_spearman_metric(all_preds, all_targets)

        return avg_loss, spearman_score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print("Loading data...")
        self.train_loader, self.val_loader, self.test_loader, self.tokenizer = (
            get_dataloaders(
                batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=Config.DEBUG
            )
        )

        best_score = -float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

        print("Starting training...")
        for epoch in range(Config.EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_score = self.validate()

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val Spearman: {val_score}"
            )

            # Early Stopping Check
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best score! Model saved.")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Load best model for prediction
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} with score {best_score}")
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
        else:
            print("No best model file found. Using current model state.")

    def predict(self):
        """
        Generates predictions for the test set and creates the submission file.
        """
        print("Generating predictions on test set...")
        self.model.eval()
        all_preds = []

        # Ensure test loader is available
        if self.test_loader is None:
            _, _, self.test_loader, _ = get_dataloaders(
                batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=Config.DEBUG
            )

        with torch.no_grad():
            for q_seq, a_seq in self.test_loader:
                q_seq = q_seq.to(self.device)
                a_seq = a_seq.to(self.device)

                outputs = self.model(q_seq, a_seq)
                all_preds.append(outputs.cpu().numpy())

        # Concatenate predictions
        # Shape: (n_samples, 30)
        predictions = np.concatenate(all_preds, axis=0)

        # Load test metadata to get qa_id
        # We assume the order in test_loader matches test.csv rows exactly
        # (shuffle=False is set in get_dataloaders for test_loader)
        test_df = pd.read_csv(Config.TEST_PATH)

        if Config.DEBUG:
            test_df = test_df.iloc[: Config.DEBUG_SIZE]

        if len(test_df) != len(predictions):
            print(
                f"Warning: Length mismatch. Test DF: {len(test_df)}, Preds: {len(predictions)}"
            )
            # Truncate to match if necessary (though shouldn't happen with correct logic)
            min_len = min(len(test_df), len(predictions))
            test_df = test_df.iloc[:min_len]
            predictions = predictions[:min_len]

        # Prepare submission DataFrame
        submission = pd.DataFrame()
        submission["qa_id"] = test_df["qa_id"]

        # Assign predicted columns
        for i, col in enumerate(Config.TARGET_COLS):
            submission[col] = predictions[:, i]

        # Save submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Validate submission format against sample
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        print(f"Submission shape: {submission.shape}")
        print(f"Sample submission shape: {sample_sub.shape}")

        # Basic check
        expected_cols = list(sample_sub.columns)
        actual_cols = list(submission.columns)
        if expected_cols == actual_cols:
            print("Submission column names match sample.")
        else:
            print("Warning: Submission column names do not match sample.")
