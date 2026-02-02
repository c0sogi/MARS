import torch
import torch.nn as nn
import numpy as np
import copy
from library.config import Config


class RNA_GRU(nn.Module):
    """
    Bidirectional GRU for RNA degradation prediction.
    Cite solution_lesson_node_00002: Preserves sequence dimension and locality.
    """

    def __init__(
        self,
        input_dim=Config.CHANNELS_PER_POS,
        hidden_dim=Config.RNN_HIDDEN_DIM,
        num_layers=Config.RNN_LAYERS,
        dropout=Config.RNN_DROPOUT,
        num_targets=Config.NUM_TARGETS,
    ):
        super(RNA_GRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Bidirectional results in 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, num_targets)

    def forward(self, x):
        # x: (Batch, Seq_Len, Channels)
        out, _ = self.gru(x)
        # out: (Batch, Seq_Len, Hidden*2)
        out = self.fc(out)
        # out: (Batch, Seq_Len, Targets)
        return out


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch (features, targets)
        features, targets = batch
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        # Slice outputs to match targets (first 68 positions)
        # outputs: (Batch, 107, 5), targets: (Batch, 68, 5)
        outputs_scored = outputs[:, : Config.SEQ_SCORED, :]

        loss = criterion(outputs_scored, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            features, targets = batch
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)

            # Slice outputs to match targets
            outputs_scored = outputs[:, : Config.SEQ_SCORED, :]

            loss = criterion(outputs_scored, targets)

            running_loss += loss.item() * features.size(0)

    return running_loss / len(loader.dataset)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
    save_path=Config.MODEL_SAVE_PATH,
):
    """
    Full training loop with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): PyTorch optimizer.
        criterion (Loss): Loss function.
        device (torch.device): Device to run on.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model weights.

    Returns:
        dict: Training history containing 'train_loss' and 'val_loss'.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": []}

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")
    return history


def predict(model, loader, device):
    """
    Generates predictions for the given data loader.

    Args:
        model (nn.Module): Trained model.
        loader (DataLoader): Data loader (test set, no targets).
        device (torch.device): Device to run on.

    Returns:
        np.ndarray: Predictions of shape (N, Output_Dim).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            # Handle case where loader returns (features, targets) or just features
            if isinstance(batch, (list, tuple)):
                features = batch[0]
            else:
                features = batch

            features = features.to(device)
            outputs = model(features)
            preds.append(outputs.cpu().numpy())

    return np.vstack(preds)
