import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model import PathologyResNet
from library.dataset import get_dataloaders


class EarlyStopping:
    """
    Early stops the training if validation AUC doesn't improve after a given patience.
    """

    def __init__(self, patience=3, verbose=False, path="checkpoint.pth"):
        """
        Args:
            patience (int): How long to wait after last time validation AUC improved.
            verbose (bool): If True, prints a message for each validation AUC improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = -np.inf

    def __call__(self, val_auc, model):
        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        """Saves model when validation AUC increases."""
        if self.verbose:
            print(
                f"Validation AUC increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_auc_max = val_auc


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions and targets for AUC calculation
        # Apply sigmoid for probability calculation, though AUC is rank-based
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case where only one class is present in the epoch (unlikely but possible)
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def predict_and_submit(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)
            outputs = model(images)
            # Apply sigmoid to get probability
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            ids_list.extend(ids)
            preds_list.extend(probs)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id": ids_list, "label": preds_list})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run():
    """
    Main execution function to run the training pipeline.
    """
    # 1. Setup
    Config.set_seed(Config.SEED)
    Config.setup_directories()
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model, Criterion, Optimizer
    model = PathologyResNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Early Stopping
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        verbose=True,
        path=Config.MODEL_SAVE_PATH,
    )

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Check early stopping
        early_stopping(val_auc, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 6. Inference
    # Load the best model saved by early_stopping
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)
