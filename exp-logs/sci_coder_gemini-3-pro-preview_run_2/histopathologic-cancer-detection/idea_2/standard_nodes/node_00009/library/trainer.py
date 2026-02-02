import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import mixup_data, mixup_criterion


class Trainer:
    """
    Manages the training and validation lifecycle of the model.
    """

    def __init__(
        self, model, optimizer, train_loader, val_loader, device=Config.DEVICE
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_auc = 0.0
        self.patience_counter = 0

    def train_epoch(self):
        """
        Runs one epoch of training with Mixup regularization.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).view(-1, 1)

            batch_size = inputs.size(0)

            # Apply Mixup if enabled
            if Config.USE_MIXUP:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, Config.MIXUP_ALPHA, self.device
                )
                outputs = self.model(inputs)
                loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lam
                )
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            count += batch_size

        return running_loss / count if count > 0 else 0.0

    def validate_epoch(self):
        """
        Runs validation and computes Loss and ROC AUC.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Apply sigmoid to get probabilities for AUC calculation
                probs = torch.sigmoid(outputs)

                running_loss += loss.item() * inputs.size(0)
                count += inputs.size(0)

                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        epoch_loss = running_loss / count if count > 0 else 0.0

        # Concatenate all batches
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Compute AUC
        try:
            epoch_auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            # Handle edge case where only one class is present in validation batch
            epoch_auc = 0.5

        return epoch_loss, epoch_auc

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {Config.NUM_EPOCHS} epochs on {self.device}...")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_epoch()
            val_loss, val_auc = self.validate_epoch()

            print(
                f"Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val AUC: {val_auc}"
            )

            # Checkpoint and Early Stopping
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT)
                print(f"New best model saved with AUC: {self.best_auc}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(f"Training complete. Best Validation AUC: {self.best_auc}")


def generate_submission(model, test_loader, device=Config.DEVICE):
    """
    Generates predictions for the test set using 8-view Test Time Augmentation (TTA).
    Saves the result to submission.csv.
    """
    print("Loading best model for inference...")
    if os.path.exists(Config.MODEL_CHECKPOINT):
        model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model weights.")

    model.eval()
    model.to(device)

    ids = []
    predictions = []

    # Retrieve IDs from the dataset dataframe directly
    # The dataset order is preserved because shuffle=False in test_loader
    test_ids = test_loader.dataset.df["id"].values

    print("Starting inference with 8-view TTA...")

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            batch_probs = torch.zeros((inputs.size(0), 1), device=device)

            # Define TTA transformations (Dihedral Group D4)
            # 1. Identity
            # 2. Rot90
            # 3. Rot180
            # 4. Rot270
            # 5. Horizontal Flip
            # 6. Horizontal Flip + Rot90
            # 7. Horizontal Flip + Rot180
            # 8. Horizontal Flip + Rot270

            # We accumulate logits or probs?
            # Averaging probabilities is standard for TTA.

            views = []

            # Original
            views.append(inputs)
            # Rotations
            views.append(torch.rot90(inputs, 1, [2, 3]))
            views.append(torch.rot90(inputs, 2, [2, 3]))
            views.append(torch.rot90(inputs, 3, [2, 3]))

            # Flip
            inputs_flip = torch.flip(inputs, [3])
            views.append(inputs_flip)
            # Flip + Rotations
            views.append(torch.rot90(inputs_flip, 1, [2, 3]))
            views.append(torch.rot90(inputs_flip, 2, [2, 3]))
            views.append(torch.rot90(inputs_flip, 3, [2, 3]))

            for view in views:
                logits = model(view)
                probs = torch.sigmoid(logits)
                batch_probs += probs

            # Average over 8 views
            batch_probs /= 8.0

            predictions.extend(batch_probs.cpu().numpy().flatten())

    # Ensure lengths match
    if len(test_ids) != len(predictions):
        print(
            f"Error: Mismatch between IDs ({len(test_ids)}) and predictions ({len(predictions)})"
        )

    # Create DataFrame
    df_submission = pd.DataFrame({"id": test_ids, "label": predictions})

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
