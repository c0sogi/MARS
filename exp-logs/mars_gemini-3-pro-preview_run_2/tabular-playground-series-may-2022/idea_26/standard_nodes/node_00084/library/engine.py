import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        # Unpack data: x_cat, x_cont, y
        x_cat, x_cont, y = data

        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        y = y.to(device).unsqueeze(1)  # Ensure target shape is (B, 1)

        optimizer.zero_grad()

        logits = model(x_cat, x_cont)
        loss = criterion(logits, y)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data in dataloader:
            x_cat, x_cont, y = data

            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(x_cat, x_cont)
            loss = criterion(logits, y)

            running_loss += loss.item()

            # Apply sigmoid to get probabilities for AUC
            probs = torch.sigmoid(logits)

            all_targets.append(y.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    avg_loss = running_loss / len(dataloader)
    auc = roc_auc_score(all_targets, all_preds)

    return avg_loss, auc


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in dataloader:
            # Test dataset returns only x_cat, x_cont
            x_cat, x_cont = data

            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)

            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)


class Trainer:
    """
    Manages the training process including logging, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        scheduler=None,
        patience=5,
        save_path=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.patience = patience
        self.save_path = save_path

        self.best_auc = 0.0
        self.patience_counter = 0

    def fit(self, train_loader, val_loader, epochs):
        print(f"Starting training for {epochs} epochs with patience {self.patience}...")

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                self.model, train_loader, self.optimizer, self.criterion, self.device
            )
            val_loss, val_auc = evaluate(
                self.model, val_loader, self.criterion, self.device
            )

            if self.scheduler:
                self.scheduler.step()

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Early Stopping and Checkpointing
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                if self.save_path:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Best AUC: {self.best_auc}"
                    )
                    break

        print(f"Training finished. Best Validation AUC: {self.best_auc}")


def generate_submission(model, test_loader, test_ids, device, output_path):
    """
    Generates predictions and saves them to a CSV file.
    """
    print("Generating predictions for submission...")
    preds = predict(model, test_loader, device)

    # Flatten predictions if necessary
    preds = preds.flatten()

    submission = pd.DataFrame({"id": test_ids, "target": preds})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
