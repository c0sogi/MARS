import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from library.config import (
    DEVICE,
    POS_WEIGHT_VAL,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    ID_COL,
    TARGET_COL,
)
from library.utils import probabilistic_f1


class Trainer:
    """
    Handles the training and validation loop for the Siamese Network.
    """

    def __init__(self, model, optimizer, scheduler, device=DEVICE, patience=3):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience

        # Initialize Loss with Aggressive Positive Weighting
        # pos_weight must be a tensor for BCEWithLogitsLoss
        weight_tensor = torch.tensor([POS_WEIGHT_VAL]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

        # Early Stopping State
        self.best_val_pf1 = -1.0
        self.early_stop_counter = 0

    def train_one_epoch(self, dataloader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        # Store predictions and targets for epoch-level metric calculation
        epoch_preds = []
        epoch_targets = []

        for batch_idx, (target_img, contra_img, labels) in enumerate(dataloader):
            # Move data to device
            target_img = target_img.to(self.device)
            contra_img = contra_img.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # Ensure shape (B, 1)

            self.optimizer.zero_grad()

            # Forward Pass: Siamese Network accepts pair of images
            logits = self.model(target_img, contra_img)

            # Loss Calculation
            loss = self.criterion(logits, labels)

            # Backward Pass
            loss.backward()

            # Note: Gradient Clipping is explicitly DISABLED for this strategy

            self.optimizer.step()

            # Metrics accumulation
            batch_size = target_img.size(0)
            running_loss += loss.item() * batch_size

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()

            epoch_preds.append(probs)
            epoch_targets.append(targets)

        # Step Scheduler (CosineAnnealing is typically stepped per epoch)
        if self.scheduler is not None:
            self.scheduler.step()

        # Aggregate Metrics
        total_samples = len(dataloader.dataset)
        avg_loss = running_loss / total_samples

        all_preds = np.concatenate(epoch_preds).flatten()
        all_targets = np.concatenate(epoch_targets).flatten()
        pf1 = probabilistic_f1(all_targets, all_preds)

        print(f"Epoch {epoch_idx} | Train Loss: {avg_loss} | Train pF1: {pf1}")

        return avg_loss, pf1

    def validate(self, dataloader, epoch_idx):
        """
        Evaluates the model on the validation set and handles early stopping.
        """
        self.model.eval()
        running_loss = 0.0

        epoch_preds = []
        epoch_targets = []

        with torch.no_grad():
            for target_img, contra_img, labels in dataloader:
                target_img = target_img.to(self.device)
                contra_img = contra_img.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                logits = self.model(target_img, contra_img)
                loss = self.criterion(logits, labels)

                batch_size = target_img.size(0)
                running_loss += loss.item() * batch_size

                probs = torch.sigmoid(logits).cpu().numpy()
                targets = labels.cpu().numpy()

                epoch_preds.append(probs)
                epoch_targets.append(targets)

        # Aggregate Metrics
        total_samples = len(dataloader.dataset)
        avg_loss = running_loss / total_samples

        all_preds = np.concatenate(epoch_preds).flatten()
        all_targets = np.concatenate(epoch_targets).flatten()
        pf1 = probabilistic_f1(all_targets, all_preds)

        print(f"Epoch {epoch_idx} | Val Loss: {avg_loss} | Val pF1: {pf1}")

        # Early Stopping & Checkpointing
        # We maximize pF1 score
        if pf1 > self.best_val_pf1:
            print(
                f"Validation pF1 improved from {self.best_val_pf1} to {pf1}. Saving model..."
            )
            self.best_val_pf1 = pf1
            self.early_stop_counter = 0
            torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
        else:
            self.early_stop_counter += 1
            print(
                f"No improvement. EarlyStopping counter: {self.early_stop_counter}/{self.patience}"
            )

        stop_training = self.early_stop_counter >= self.patience
        return avg_loss, pf1, stop_training


def predict(model, dataloader, device=DEVICE):
    """
    Generates predictions for the test set, aggregates them by prediction_id,
    and saves the submission file.
    """
    # 1. Load Best Model
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading best model weights from {MODEL_SAVE_PATH}...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print(
            "Warning: No checkpoint found. Using current model weights for inference."
        )

    model.to(device)
    model.eval()

    all_probs = []

    # 2. Inference Loop
    print("Starting inference on test set...")
    with torch.no_grad():
        for target_img, contra_img, _ in dataloader:
            target_img = target_img.to(device)
            contra_img = contra_img.to(device)

            logits = model(target_img, contra_img)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)

    # Flatten predictions
    flat_probs = np.concatenate(all_probs).flatten()

    # 3. Map Predictions to Metadata
    # Access the dataframe from the dataset
    df_test = dataloader.dataset.df.copy()

    if len(df_test) != len(flat_probs):
        raise ValueError(
            f"Mismatch: {len(df_test)} metadata rows vs {len(flat_probs)} predictions."
        )

    df_test[TARGET_COL] = flat_probs

    # 4. Aggregation
    # Group by 'prediction_id' and take the MAX probability.
    # This handles cases where multiple views (CC/MLO) map to the same breast.
    submission_df = df_test.groupby(ID_COL)[TARGET_COL].max().reset_index()

    # 5. Save Submission
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
    print(submission_df.head())
