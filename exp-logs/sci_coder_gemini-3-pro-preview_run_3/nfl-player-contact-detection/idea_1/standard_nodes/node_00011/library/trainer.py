import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from library.config import (
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
    setup_reproducibility,
)
from library.model import ContactMLP
from library.dataset import get_dataloaders

# Ensure reproducibility
setup_reproducibility(SEED)


def compute_mcc(y_true, y_pred_prob, threshold=0.5):
    """
    Computes the Matthews Correlation Coefficient.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred_prob (np.ndarray): Predicted probabilities.
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        float: The MCC score.
    """
    y_pred = (y_pred_prob > threshold).astype(int)
    return matthews_corrcoef(y_true, y_pred)


class Trainer:
    """
    Manages the training, validation, and early stopping of the ContactMLP model.
    """

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_mcc = -1.0

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0

        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device).unsqueeze(1)  # Match output shape (batch, 1)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def evaluate(self, loader):
        """Evaluates the model on a given loader."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device).unsqueeze(1)

                outputs = self.model(X)
                loss = self.criterion(outputs, y)

                running_loss += loss.item() * X.size(0)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        avg_loss = running_loss / len(loader.dataset)
        # Use default 0.5 threshold for monitoring during training
        mcc = compute_mcc(all_labels, all_preds, threshold=0.5)

        return avg_loss, mcc, all_labels, all_preds

    def fit(self, num_epochs, early_stopping_patience, model_save_path):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training on {self.device}...")
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_mcc, _, _ = self.evaluate(self.val_loader)

            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val MCC: {val_mcc}"
            )

            # Early Stopping Logic based on MCC
            if val_mcc > self.best_mcc:
                self.best_mcc = val_mcc
                patience_counter = 0
                torch.save(self.model.state_dict(), model_save_path)
                print(f"New best model saved to {model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training finished. Best Validation MCC: {self.best_mcc}")


def optimize_threshold(model, val_loader, device):
    """
    Finds the probability threshold that maximizes MCC on the validation set.
    """
    print("Optimizing threshold on validation set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            outputs = model(X)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    best_threshold = 0.5
    best_mcc = -1.0

    # Search range from 0.1 to 0.9
    thresholds = np.arange(0.1, 0.95, 0.05)
    for thresh in thresholds:
        mcc = compute_mcc(all_labels, all_preds, threshold=thresh)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh

    print(f"Optimal Threshold: {best_threshold} (MCC: {best_mcc})")
    return best_threshold


def generate_submission(model, test_loader, test_ids, threshold, output_path, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            outputs = model(X)
            all_preds.append(outputs.cpu().numpy())

    all_preds = np.concatenate(all_preds)

    # Apply optimized threshold
    predictions = (all_preds > threshold).astype(int).flatten()

    # Create DataFrame matching sample_submission format
    df_sub = pd.DataFrame({"contact_id": test_ids, "contact": predictions})

    # Save to CSV
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_pipeline():
    """
    Main pipeline to load data, train model, optimize threshold, and generate submission.
    """
    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders()

    # 2. Initialize Model
    # Get input dimension from the dataset (number of features)
    input_dim = train_loader.dataset.X.shape[1]
    model = ContactMLP(input_dim=input_dim).to(DEVICE)

    # 3. Setup Training Components
    # Using BCELoss as the model outputs probabilities via Sigmoid
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
    )

    # 4. Train Model
    trainer.fit(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        model_save_path=MODEL_SAVE_PATH,
    )

    # 5. Load Best Model for Inference
    print(f"Loading best model from {MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    # 6. Optimize Threshold
    best_threshold = optimize_threshold(model, val_loader, DEVICE)

    # 7. Generate Submission
    generate_submission(
        model, test_loader, test_ids, best_threshold, SUBMISSION_PATH, DEVICE
    )
