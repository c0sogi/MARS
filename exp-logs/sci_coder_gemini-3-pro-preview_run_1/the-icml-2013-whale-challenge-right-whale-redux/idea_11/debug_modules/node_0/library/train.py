import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import calculate_auc, Mixup
from library.dataset import get_dataloaders
from library.model import HierarchicalCRNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup and Gradient Clipping.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = Mixup(inputs, targets, Config.MIXUP_ALPHA)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)

        # Compute Mixup Loss
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
            outputs, targets_b
        )

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store predictions and targets for AUC
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    epoch_auc = calculate_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Handle scalar output for batch size 1
            if np.ndim(probs) == 0:
                probs = [probs]

            all_preds.extend(probs)

    return all_preds


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Handle Class Imbalance
        pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        lr = learning_rate if learning_rate is not None else Config.LEARNING_RATE
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

        self.best_auc = 0.0
        self.patience_counter = 0
        self.best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

        # Ensure cache dir exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def fit(self, n_epochs=None):
        epochs = n_epochs if n_epochs is not None else Config.N_EPOCHS
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
            )
            val_loss, val_auc = validate_one_epoch(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Print full precision as requested
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            self.scheduler.step(val_auc)

            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                self.patience_counter = 0
                print("New best model saved.")
            else:
                self.patience_counter += 1

            # Early Stopping
            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val AUC: {self.best_auc}")


def run_training(n_epochs=None, load_cached_data=True):
    """
    Orchestrates the training process: loads data, trains model, and generates submission.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    print("Initializing HierarchicalCRNN model...")
    model = HierarchicalCRNN().to(device)

    # 3. Train
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit(n_epochs=n_epochs)

    # 4. Inference
    print("Loading best model for inference...")
    if os.path.exists(trainer.best_model_path):
        model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))
    else:
        print("Warning: Best model not found, using current model state.")

    print("Generating predictions on test set...")
    predictions = predict(model, test_loader, device)

    # 5. Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Retrieve test IDs from the dataset attached to the loader
    test_ids = test_loader.dataset.ids

    submission_df = pd.DataFrame({"clip": test_ids, "probability": predictions})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head().to_string())
