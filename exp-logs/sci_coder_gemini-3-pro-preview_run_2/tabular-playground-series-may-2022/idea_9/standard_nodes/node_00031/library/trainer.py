import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, compute_auc


class Trainer:
    """
    Trainer class for the Hybrid Attention-ResFunnel-GLU model.
    Handles training loop, validation, early stopping, and submission generation.
    """

    def __init__(self, model, device=None):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (torch.device, optional): Device to run training on.
        """
        self.device = device if device else torch.device(Config.DEVICE)
        self.model = model.to(self.device)

        # Optimization Setup
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=Config.LR_SCHEDULER_STEP_SIZE,
            gamma=Config.LR_SCHEDULER_GAMMA,
        )

        # Loss Function (Binary Cross Entropy with Logits)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Move data to device
            cont_features = batch["cont_features"].to(self.device)
            cat_sequence = batch["cat_sequence"].to(self.device)
            target = batch["target"].to(self.device).unsqueeze(1)  # Shape (B, 1)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(cont_features, cat_sequence)
            loss = self.criterion(logits, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average loss and AUC.
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in val_loader:
                cont_features = batch["cont_features"].to(self.device)
                cat_sequence = batch["cat_sequence"].to(self.device)
                target = batch["target"].to(self.device).unsqueeze(1)

                logits = self.model(cont_features, cat_sequence)
                loss = self.criterion(logits, target)
                total_loss += loss.item()

                # Apply sigmoid to get probabilities for AUC calculation
                probs = torch.sigmoid(logits)

                all_targets.append(target.cpu())
                all_preds.append(probs.cpu())

        # Compute Metrics
        avg_loss = total_loss / len(val_loader)

        # Concatenate all batches
        y_true = torch.cat(all_targets)
        y_pred = torch.cat(all_preds)

        auc_score = compute_auc(y_true, y_pred)

        return avg_loss, auc_score

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping based on Validation AUC.
        """
        print(f"Starting training on {self.device}...")
        best_auc = 0.0
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_auc = self.validate(val_loader)

            # Update Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch + 1}/{Config.EPOCHS} | LR: {current_lr} | "
                f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Early Stopping & Model Checkpointing
            # Decoupled from loss: strictly maximize AUC
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"New best model saved to {Config.MODEL_PATH} (AUC: {best_auc})")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered. No improvement in AUC for {Config.PATIENCE} epochs."
                    )
                    break

        print(f"Training complete. Best Validation AUC: {best_auc}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        # Load best model weights
        if os.path.exists(Config.MODEL_PATH):
            state_dict = torch.load(Config.MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded best model weights from {Config.MODEL_PATH}")
        else:
            print("Warning: No saved model found. Using current model state.")

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                cont_features = batch["cont_features"].to(self.device)
                cat_sequence = batch["cat_sequence"].to(self.device)

                logits = self.model(cont_features, cat_sequence)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())

        # Flatten list of arrays
        return np.concatenate(all_probs).flatten()

    def generate_submission(self, test_loader, test_ids):
        """
        Generates predictions and saves them to the submission file.
        """
        print("Generating predictions for test set...")
        predictions = self.predict(test_loader)

        if len(predictions) != len(test_ids):
            raise ValueError(
                f"Mismatch: {len(predictions)} predictions vs {len(test_ids)} test IDs."
            )

        # Create DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

        # Save to disk
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
