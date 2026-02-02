import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy
from scipy.stats import spearmanr
from library.config import Config
from library.dataset import get_dataloaders
from library.network import DualGRUNet


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_spearman_metric(preds, targets):
    """
    Computes Mean column-wise Spearman's correlation coefficient.
    Args:
        preds: Numpy array of shape (N, 30)
        targets: Numpy array of shape (N, 30)
    Returns:
        float: Mean Spearman correlation
    """
    corrs = []
    for i in range(preds.shape[1]):
        # Handle constant columns to avoid NaNs
        if np.std(preds[:, i]) == 0 or np.std(targets[:, i]) == 0:
            corr = 0.0
        else:
            res = spearmanr(preds[:, i], targets[:, i])
            corr = res.statistic

        if np.isnan(corr):
            corr = 0.0
        corrs.append(corr)

    return np.mean(corrs)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for q, a, y in loader:
        q = q.to(device)
        a = a.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(q, a)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * q.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns:
        val_loss: Average BCE Loss
        spearman_score: Mean column-wise Spearman correlation
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for q, a, y in loader:
            q = q.to(device)
            a = a.to(device)
            y = y.to(device)

            outputs = model(q, a)
            loss = criterion(outputs, y)

            running_loss += loss.item() * q.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    # Concatenate for metric calculation
    preds_arr = np.vstack(all_preds)
    targets_arr = np.vstack(all_targets)

    spearman_score = compute_spearman_metric(preds_arr, targets_arr)

    return val_loss, spearman_score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for q, a, _ in loader:
            q = q.to(device)
            a = a.to(device)

            outputs = model(q, a)
            all_preds.append(outputs.cpu().numpy())

    return np.vstack(all_preds)


class Trainer:
    """
    Trainer class to encapsulate training, validation, and prediction logic.
    """

    def __init__(self, load_cached_data=True, debug_sample_size=None):
        self.load_cached_data = load_cached_data
        self.debug_sample_size = debug_sample_size
        self.device = Config.DEVICE
        set_seed(Config.SEED)

        # Initialize Data
        # Using get_dataloaders from library.dataset which handles caching and preprocessing
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.vocab,
            self.test_ids,
        ) = get_dataloaders(
            batch_size=Config.BATCH_SIZE,
            load_cached_data=self.load_cached_data,
            debug_sample_size=self.debug_sample_size,
        )

        # Initialize Model
        self.model = DualGRUNet().to(self.device)

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.BCELoss()

    def run(self, epochs=Config.EPOCHS, patience=Config.PATIENCE):
        """
        Executes the training pipeline with Early Stopping.
        """
        print(f"Starting training on {self.device}...")

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Train
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
            )

            # Validate
            val_loss, val_spearman = validate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Spearman: {val_spearman}"
            )

            # Early Stopping Check (Monitor Val Loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Cite solution_lesson_node_00001: Deep copy to save actual weights, not reference
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

        # Load best weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with Val Loss: {best_val_loss}")

    def generate_submission(self):
        """
        Generates and saves the submission file.
        """
        print("Generating predictions for test set...")
        predictions = predict(self.model, self.test_loader, self.device)

        submission = pd.DataFrame(predictions, columns=Config.TARGET_COLS)
        submission.insert(0, "qa_id", self.test_ids)

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
