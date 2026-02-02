import torch
import torch.nn as nn
import numpy as np
import copy
from library.config import Config


class GlobalMLP(nn.Module):
    """
    Global Feedforward Perceptron (MLP) for RNA degradation prediction.

    Architecture:
    - Input Layer: Flattened feature vector (1498 dimensions).
    - Hidden Layers: Stack of Dense -> BatchNorm -> ReLU -> Dropout blocks.
    - Output Layer: Dense (Linear) projection to flattened targets (340 dimensions).
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        output_dim=Config.OUTPUT_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Initializes the GlobalMLP model.

        Args:
            input_dim (int): Size of the input feature vector.
            output_dim (int): Size of the output target vector.
            hidden_layers (list): List of integers defining the size of each hidden layer.
            dropout_rate (float): Probability of an element to be zeroed in Dropout.
        """
        super(GlobalMLP, self).__init__()

        layers = []
        current_dim = input_dim

        # Construct Hidden Layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # Output Layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.model(x)


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
        loss = criterion(outputs, targets)

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
            loss = criterion(outputs, targets)

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
