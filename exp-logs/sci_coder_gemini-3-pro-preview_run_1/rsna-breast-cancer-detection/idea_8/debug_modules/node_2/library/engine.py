import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import probabilistic_f1
from library.model import SymmetryDifferenceSiameseNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, labels in loader:
        # Inputs are tuples: (target_image, contralateral_image)
        target_img = inputs[0].to(device, non_blocking=True)
        contra_img = inputs[1].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        logits = model((target_img, contra_img))
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient Clipping is DISABLED as per strategy to allow large updates for minority class
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (Average Loss, Probabilistic F1 Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            target_img = inputs[0].to(device, non_blocking=True)
            contra_img = inputs[1].to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            logits = model((target_img, contra_img))
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate results for global metric calculation
    y_pred = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    pf1 = probabilistic_f1(y_true, y_pred)

    return epoch_loss, pf1


class EarlyStopping:
    """
    Implements early stopping based on validation pF1 score.
    """

    def __init__(self, patience=3, min_delta=0.0, path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        # Score is pF1 (higher is better)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS, device=None):
    """
    Main training loop with Early Stopping.

    Args:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        epochs: Maximum number of epochs.
        device: Torch device.

    Returns:
        str: Path to the best saved model.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    print(f"Initializing model on {device}...")
    model = SymmetryDifferenceSiameseNet().to(device)

    # Loss with positive weight to handle class imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    early_stopping = EarlyStopping(patience=3, path=best_model_path)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pf1 = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Printing full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val pF1: {val_pf1}"
        )

        early_stopping(val_pf1, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best pF1: {early_stopping.best_score}")
    return best_model_path


def predict_and_submit(model_path, test_loader, device=None):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model_path: Path to the saved model weights.
        test_loader: DataLoader for test data.
        device: Torch device.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    print(f"Loading model from {model_path}...")
    model = SymmetryDifferenceSiameseNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    print("Running inference on test set...")

    with torch.no_grad():
        for inputs, prediction_ids in test_loader:
            target_img = inputs[0].to(device, non_blocking=True)
            contra_img = inputs[1].to(device, non_blocking=True)

            logits = model((target_img, contra_img))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Map predictions back to prediction_ids
            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Aggregate predictions: Max probability per prediction_id (handling multiple views)
    df = pd.DataFrame(results)
    submission = df.groupby("prediction_id")["cancer"].max().reset_index()

    submission_path = Config.SUBMISSION_PATH
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
