import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.model import ResNet18Baseline
from library.utils import seed_everything


class Trainer:
    """
    Trainer class for the ResNet18 Baseline model.
    Handles training, validation, early stopping, and submission generation.
    """

    def __init__(self):
        """
        Initialize the Trainer with model, optimizer, loss function, and scheduler.
        """
        # Set seed for reproducibility
        seed_everything(Config.SEED)

        self.device = Config.DEVICE

        # Initialize model
        # Pretrained weights are used as per the baseline design
        self.model = ResNet18Baseline(pretrained=True).to(self.device)

        # Binary Cross Entropy Loss with Logits
        # Handles numerical stability and expects logits from the model
        self.criterion = nn.BCEWithLogitsLoss()

        # AdamW Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Step Learning Rate Scheduler
        # Decays LR by gamma every step_size epochs
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.9
        )

        # Early Stopping State
        self.patience = Config.EARLY_STOPPING_PATIENCE
        self.min_delta = Config.EARLY_STOPPING_MIN_DELTA
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def train_one_epoch(self, train_loader, max_batches=None):
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            max_batches (int, optional): Limit number of batches for debugging.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for i, (images, labels) in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # Shape: (Batch, 1)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        return running_loss / total_samples if total_samples > 0 else 0.0

    def validate_epoch(self, val_loader, max_batches=None):
        """
        Evaluates the model on the validation set.
        Applies probability calibration to logits before loss calculation.
        Cite solution_lesson_node_00001: Impact of Balanced Sampling on Probabilistic Calibration.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
            max_batches (int, optional): Limit number of batches for debugging.

        Returns:
            float: Average validation loss for the epoch.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        # Calculate calibration offset
        # Shift logits by log(odds_test) - log(odds_train)
        # odds = p / (1 - p)
        prob_train = Config.POSITIVE_SAMPLING_RATIO
        prob_test = Config.NATURAL_PREVALENCE

        logit_offset = np.log(prob_test / (1 - prob_test)) - np.log(
            prob_train / (1 - prob_train)
        )
        logit_offset = torch.tensor(
            logit_offset, device=self.device, dtype=torch.float32
        )

        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                if max_batches is not None and i >= max_batches:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                # Get raw logits
                logits = self.model(images)

                # Apply calibration
                corrected_logits = logits + logit_offset

                # Compute loss on corrected logits against natural distribution labels
                loss = self.criterion(corrected_logits, labels)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

        return running_loss / total_samples if total_samples > 0 else 0.0

    def fit(self, train_loader, val_loader, epochs=Config.NUM_EPOCHS, max_batches=None):
        """
        Runs the full training loop with early stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            max_batches (int, optional): Limit batches per epoch for debugging.
        """
        print(f"Starting training on {self.device} for {epochs} epochs.")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, max_batches=max_batches)
            val_loss = self.validate_epoch(val_loader, max_batches=max_batches)

            # Print full precision as requested
            print(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss}")

            self.scheduler.step()

            # Early Stopping Logic
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.early_stop_counter = 0

                # Save best model
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
                print(
                    f"Validation loss improved. Model saved to {Config.MODEL_CHECKPOINT_PATH}"
                )
            else:
                self.early_stop_counter += 1
                print(
                    f"No improvement. Early stopping counter: {self.early_stop_counter}/{self.patience}"
                )

            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered. Training stopped.")
                break

    def generate_submission(self, test_loader, output_path=Config.SUBMISSION_PATH):
        """
        Generates predictions for the test set and saves the submission CSV.
        Uses Max Pooling to aggregate image-level probabilities to prediction_id level.

        Args:
            test_loader (DataLoader): Test data.
            output_path (str): Path to save the submission CSV.
        """
        # Load best weights if available
        if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=self.device)
            )
            print(f"Loaded best model weights from {Config.MODEL_CHECKPOINT_PATH}")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()

        all_probs = []
        all_ids = []

        print("Generating predictions...")

        # Calculate calibration offset
        # Cite solution_lesson_node_00001
        prob_train = Config.POSITIVE_SAMPLING_RATIO
        prob_test = Config.NATURAL_PREVALENCE
        logit_offset = np.log(prob_test / (1 - prob_test)) - np.log(
            prob_train / (1 - prob_train)
        )
        logit_offset = torch.tensor(
            logit_offset, device=self.device, dtype=torch.float32
        )

        with torch.no_grad():
            for images, pred_ids in test_loader:
                images = images.to(self.device)

                # Forward pass (returns logits)
                logits = self.model(images)

                # Apply calibration
                corrected_logits = logits + logit_offset

                # Sigmoid to get probabilities
                probs = torch.sigmoid(corrected_logits).cpu().numpy().flatten()

                all_probs.extend(probs)
                all_ids.extend(pred_ids)

        # Create DataFrame
        df_pred = pd.DataFrame({"prediction_id": all_ids, "cancer": all_probs})

        # Max Pooling Aggregation: Group by prediction_id and take max probability
        # This assumes that if any view shows cancer, the breast is positive.
        submission_df = df_pred.groupby("prediction_id", as_index=False)["cancer"].max()

        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
