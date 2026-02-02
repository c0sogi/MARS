import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.model import CRNN


class Trainer:
    """
    Manages the training, validation, and inference processes for the Right Whale Detection model.
    """

    def __init__(self):
        self.device = Config.DEVICE
        set_seed(Config.SEED)
        Config.setup_dirs()

    def train_epoch(self, model, loader, criterion, optimizer):
        """
        Runs one epoch of training.
        """
        model.train()
        running_loss = 0.0
        all_targets = []
        all_scores = []

        for batch in loader:
            # Move data to device
            inputs = batch["data"].to(self.device)
            targets = batch["label"].to(self.device).unsqueeze(1)  # Shape: (B, 1)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)

            # Compute loss
            loss = criterion(logits, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_scores.extend(probs)
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)
        epoch_auc = calculate_roc_auc(np.array(all_targets), np.array(all_scores))

        return epoch_loss, epoch_auc

    def validate(self, model, loader, criterion):
        """
        Runs validation on the provided loader.
        """
        model.eval()
        running_loss = 0.0
        all_targets = []
        all_scores = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["data"].to(self.device)
                targets = batch["label"].to(self.device).unsqueeze(1)

                logits = model(inputs)
                loss = criterion(logits, targets)

                running_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_scores.extend(probs)
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)
        epoch_auc = calculate_roc_auc(np.array(all_targets), np.array(all_scores))

        return epoch_loss, epoch_auc

    def fit(self, train_loader, val_loader, epochs=Config.NUM_EPOCHS):
        """
        Main training loop with Early Stopping and Scheduler.
        """
        print(f"Initializing model on {self.device}...")
        model = CRNN().to(self.device)

        # Handle Class Imbalance
        pos_weight = torch.tensor([Config.POS_WEIGHT]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=Config.LR_SCHEDULER_FACTOR,
            patience=Config.LR_SCHEDULER_PATIENCE,
        )

        best_val_auc = 0.0
        patience_counter = 0

        print("Starting training...")

        for epoch in range(epochs):
            train_loss, train_auc = self.train_epoch(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_auc = self.validate(model, val_loader, criterion)

            # Print full precision metrics
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss}, Train AUC: {train_auc}, "
                f"Val Loss: {val_loss}, Val AUC: {val_auc}"
            )

            # Update Scheduler
            scheduler.step(val_auc)

            # Checkpoint and Early Stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation AUC: {best_val_auc}")
        return best_val_auc

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Generating predictions...")

        # Load Model
        model = CRNN().to(self.device)
        if os.path.exists(Config.MODEL_SAVE_PATH):
            model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: No checkpoint found. Using initialized weights.")

        model.eval()
        predictions = []
        clip_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["data"].to(self.device)
                ids = batch["id"]  # List of clip IDs

                logits = model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                predictions.extend(probs)
                clip_ids.extend(ids)

        # Create Submission DataFrame
        df = pd.DataFrame({"clip": clip_ids, "probability": predictions})

        # Save to CSV
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
