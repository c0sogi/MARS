import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, save_best_model
from library.model import SAHCN


class Trainer:
    """
    Manages the training, validation, and prediction processes for the SAHCN model.
    """

    def __init__(self):
        """
        Initializes the Trainer with model, optimizer, criterion, and scheduler.
        """
        # Set seed for reproducibility
        set_seed(Config.SEED)

        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = SAHCN().to(self.device)

        # Loss Function: BCELoss because model output is Sigmoid
        self.criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.LR_FACTOR,
            patience=Config.LR_PATIENCE,
            min_lr=Config.MIN_LR,
        )

    def train_one_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Move data to device
            images = batch["image"].to(self.device)
            inc_angles = batch["inc_angle"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1)  # (Batch, 1)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, inc_angles)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation on the given loader.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                inc_angles = batch["inc_angle"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1)

                outputs = self.model(images, inc_angles)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            # Train and Validate
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Scheduler Step
            # Note: ReduceLROnPlateau expects the metric (val_loss)
            self.scheduler.step(val_loss)

            # Get current LR for logging
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"LR: {current_lr} - "
                f"Time: {elapsed}s"
            )

            # Early Stopping and Model Checkpointing
            if val_loss < best_val_loss:
                print(
                    f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
                )
                best_val_loss = val_loss
                patience_counter = 0
                save_best_model(self.model, Config.MODEL_PATH)
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
                )

                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Loss: {best_val_loss}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        Saves the submission file to Config.SUBMISSION_FILE.
        """
        print("Loading best model for prediction...")

        # Load the best model weights
        if os.path.exists(Config.MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_PATH, map_location=self.device)
            )
        else:
            print("Warning: Model checkpoint not found. Using current model state.")

        self.model.eval()

        ids = []
        predictions = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                inc_angles = batch["inc_angle"].to(self.device)
                batch_ids = batch["id"]

                outputs = self.model(images, inc_angles)

                # outputs are probabilities (Sigmoid applied in model)
                probs = outputs.cpu().numpy().flatten()

                ids.extend(batch_ids)
                predictions.extend(probs)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": ids, "is_iceberg": predictions})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        print(submission_df.head())
