import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library import config, model_mlp


class EarlyStopping:
    """
    Early stops the training if validation AUC doesn't improve after a given patience.
    """

    def __init__(self, patience=config.MLP_PATIENCE, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = -np.inf
        self.best_model_state = None

    def __call__(self, val_auc, model):
        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        """Saves model state when validation AUC increases."""
        if self.verbose:
            print(
                f"Validation AUC increased ({self.val_auc_max} --> {val_auc}).  Saving model state..."
            )
        self.best_model_state = model.state_dict()
        self.val_auc_max = val_auc


def train_mlp_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for batch_idx, (data, targets) in enumerate(loader):
        # Move inputs to device
        title = data["title_emb"].to(device)
        body = data["body_emb"].to(device)
        history = data["history_seq"].to(device)
        history_mask = data["history_mask"].to(device)
        persona = data["persona_centroid"].to(device)
        metadata = data["dense_metadata"].to(device)

        targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(title, body, history, history_mask, persona, metadata)

        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

        # Store predictions for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case with single class in batch/dataset
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate_mlp(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            title = data["title_emb"].to(device)
            body = data["body_emb"].to(device)
            history = data["history_seq"].to(device)
            history_mask = data["history_mask"].to(device)
            persona = data["persona_centroid"].to(device)
            metadata = data["dense_metadata"].to(device)

            targets = targets.to(device).unsqueeze(1)

            logits = model(title, body, history, history_mask, persona, metadata)
            loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs)

    val_loss = running_loss / len(loader.dataset)
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(
    train_loader, val_loader, meta_dim, device=None, epochs=config.MLP_EPOCHS
):
    """
    Manages the full training lifecycle including initialization, training loop,
    validation, and early stopping.

    Args:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        meta_dim: Dimension of the dense metadata features.
        device: Torch device (optional).
        epochs: Number of epochs to train.

    Returns:
        model: The trained model with the best state loaded.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting MLP training on {device}...")

    # Initialize Model
    model = model_mlp.PersonaAwareSkipGatedMLP(meta_dim=meta_dim)
    model.to(device)

    # Optimizer & Criterion
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.MLP_LEARNING_RATE,
        weight_decay=config.MLP_WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping
    early_stopping = EarlyStopping(patience=config.MLP_PATIENCE, verbose=True)

    for epoch in range(epochs):
        train_loss, train_auc = train_mlp_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate_mlp(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss}, Train AUC: {train_auc}, "
            f"Val Loss: {val_loss}, Val AUC: {val_auc}"
        )

        early_stopping(val_auc, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model state
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        print("Loaded best model from early stopping checkpoint.")

    return model
