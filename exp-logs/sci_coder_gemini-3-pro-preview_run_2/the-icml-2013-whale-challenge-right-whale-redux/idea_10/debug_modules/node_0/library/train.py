import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import (
    WORK_DIR,
    SUBMISSION_FILE,
    DEVICE,
    NUM_EPOCHS,
    PATIENCE,
    LR,
    WEIGHT_DECAY,
    N_FOLDS,
    SEED,
)
from library.utils import seed_everything, get_logger
from library.model import get_model
from library.data import get_dataloaders


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    Saves the model checkpoint.
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pth",
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        """Saves model when validation score increases."""
        if self.verbose:
            self.trace_func(
                f"Validation score increased ({self.val_score_max:.6f} --> {val_score:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_score_max = val_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5  # Handle edge case with single class in batch/set

    return epoch_loss, auc


def inference(model, loader, device):
    model.eval()
    all_probs = []
    all_clips = []

    with torch.no_grad():
        for inputs, clips in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_clips.extend(clips)

    all_probs = np.concatenate(all_probs)
    return all_clips, all_probs


def run_fold(fold, logger):
    logger.info(f"Starting Fold {fold}")

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(fold=fold, load_cached_data=True)

    # Initialize Model
    model = get_model()
    model.to(DEVICE)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Early Stopping
    checkpoint_path = os.path.join(WORK_DIR, f"model_fold_{fold}.pth")
    early_stopping = EarlyStopping(
        patience=PATIENCE, verbose=False, path=checkpoint_path
    )

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        elapsed = time.time() - start_time

        logger.info(
            f"Fold {fold} | Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.15f} | Time: {elapsed:.2f}s"
        )

        early_stopping(val_auc, model)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    logger.info(f"Fold {fold} Best AUC: {early_stopping.best_score:.15f}")
    return checkpoint_path


def train_and_predict():
    seed_everything(SEED)
    logger = get_logger()

    # Train Folds
    model_paths = []
    for fold in range(N_FOLDS):
        path = run_fold(fold, logger)
        model_paths.append(path)

    # Inference and Ensemble
    logger.info("Starting Inference and Ensemble...")

    # We only need one test loader since test data is the same for all folds
    _, _, test_loader = get_dataloaders(fold=0, load_cached_data=True)

    fold_preds = []
    test_clips = None

    for fold, path in enumerate(model_paths):
        logger.info(f"Predicting with model from Fold {fold}")
        model = get_model()
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)

        clips, probs = inference(model, test_loader, DEVICE)

        if test_clips is None:
            test_clips = clips

        # probs shape is (N, 1), flatten to (N,)
        fold_preds.append(probs.flatten())

    # Average predictions (Soft Voting)
    avg_preds = np.mean(fold_preds, axis=0)

    # Create Submission
    submission_df = pd.DataFrame({"clip": test_clips, "probability": avg_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

    submission_df.to_csv(SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {SUBMISSION_FILE}")
