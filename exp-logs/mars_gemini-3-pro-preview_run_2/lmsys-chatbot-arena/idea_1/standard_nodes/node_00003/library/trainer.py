import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger

logger = get_logger("trainer")


class ModelTrainer:
    """
    Manages the training, validation, and inference processes for the Chatbot Preference model.
    """

    def __init__(self, model, train_loader, val_loader, test_loader):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            test_loader (DataLoader): DataLoader for test data.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = Config.DEVICE

        # Move model to device
        self.model.to(self.device)

        # Loss function: CrossEntropyLoss expects logits and class indices
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

    def train_epoch(self):
        """
        Runs one epoch of training.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for features, targets in self.train_loader:
            # Handle dictionary inputs
            if isinstance(features, dict):
                features = {k: v.to(self.device) for k, v in features.items()}
            else:
                features = features.to(self.device)

            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(features)
            loss = self.criterion(logits, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Accumulate loss (multiply by batch size to handle last batch correctly)
            # If features is dict, get batch size from one value
            if isinstance(features, dict):
                batch_size = next(iter(features.values())).size(0)
            else:
                batch_size = features.size(0)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        with torch.no_grad():
            for features, targets in self.val_loader:
                if isinstance(features, dict):
                    features = {k: v.to(self.device) for k, v in features.items()}
                    batch_size = next(iter(features.values())).size(0)
                else:
                    features = features.to(self.device)
                    batch_size = features.size(0)

                targets = targets.to(self.device)

                logits = self.model(features)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

        avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return avg_loss

    def train(self):
        """
        Main training loop with Early Stopping.
        """
        logger.info("Starting training...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Print full precision as requested
            logger.info(
                f"Epoch {epoch + 1}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                logger.info(
                    f"Validation loss improved. Model saved to {Config.MODEL_SAVE_PATH}"
                )
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training complete. Best Validation Loss: {best_val_loss}")

    def generate_submission(self):
        """
        Generates predictions for the test set and saves them to a CSV file.
        """
        logger.info("Generating submission...")

        # Load the best model weights if available
        if os.path.exists(Config.MODEL_SAVE_PATH):
            logger.info(f"Loading best model from {Config.MODEL_SAVE_PATH}")
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            logger.warning(
                "No checkpoint found. Using current model weights for inference."
            )

        self.model.eval()
        all_probs = []

        # Inference loop
        with torch.no_grad():
            for features in self.test_loader:
                if isinstance(features, dict):
                    features = {k: v.to(self.device) for k, v in features.items()}
                else:
                    features = features.to(self.device)

                # Forward pass
                logits = self.model(features)

                # Apply Softmax to get probabilities
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        # Concatenate all batches
        if len(all_probs) > 0:
            all_probs = np.concatenate(all_probs, axis=0)
        else:
            all_probs = np.array([])

        # Load test metadata to get IDs
        # Note: We must ensure alignment. The test_loader iterates sequentially over test.csv.
        test_df = pd.read_csv(Config.TEST_DATA_PATH)

        # Handle Debug mode: if debug was on, the loader only processed a subset.
        # We must truncate the dataframe to match the predictions.
        if Config.DEBUG:
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

        if len(test_df) != len(all_probs):
            logger.error(
                f"Mismatch between test IDs ({len(test_df)}) and predictions ({len(all_probs)})."
            )
            # Fallback to intersection length to avoid crash, though this indicates an issue
            min_len = min(len(test_df), len(all_probs))
            test_df = test_df.iloc[:min_len]
            all_probs = all_probs[:min_len]

        # Create submission DataFrame
        # Column mapping based on data_processing.py target encoding:
        # 0: winner_model_a, 1: winner_model_b, 2: winner_tie
        submission = pd.DataFrame(
            {
                "id": test_df["id"],
                "winner_model_a": all_probs[:, 0],
                "winner_model_b": all_probs[:, 1],
                "winner_tie": all_probs[:, 2],
            }
        )

        # Save to CSV
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
