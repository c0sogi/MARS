import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from library.utils import compute_metrics, inverse_transform_target


class ResidualBlock(nn.Module):
    """
    A Residual Block consisting of Linear -> BN -> ReLU -> Dropout -> Linear
    with a skip connection.
    """

    def __init__(self, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetMLP(nn.Module):
    """
    Deep Residual Feedforward Neural Network.
    Structure: Input Projection -> Stack of Residual Blocks -> Output Head.
    """

    def __init__(
        self, input_dim, num_classes, num_blocks=3, hidden_dim=256, dropout_rate=0.2
    ):
        super(ResNetMLP, self).__init__()

        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        # Stack of residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        # Output classification head
        self.output_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        logits = self.output_head(out)
        return logits


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Get predictions for metrics
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_acc = compute_metrics(all_targets, all_preds)

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    epoch_acc = compute_metrics(all_targets, all_preds)

    return epoch_loss, epoch_acc


def train_model(
    model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5, device="cuda"
):
    """
    Trains the ResNetMLP model with Early Stopping.
    """
    print(f"Starting training on device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, "
            f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights.")

    return model


def predict_and_submit(
    model, test_loader, output_path="./submission/submission.csv", device="cuda"
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    print("Generating predictions...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu().numpy())

    # Flatten predictions
    all_preds = np.concatenate(all_preds)

    # Convert 0-indexed predictions back to original class labels
    final_preds = inverse_transform_target(all_preds)

    # Load Test IDs from cache (created by data_loader.py)
    # Assuming data_loader.py saves it to ./working/idea_2/test_ids.npy
    cache_id_path = "./working/idea_2/test_ids.npy"
    if not os.path.exists(cache_id_path):
        raise FileNotFoundError(
            f"Test IDs not found at {cache_id_path}. Ensure data_loader is run first."
        )

    test_ids = np.load(cache_id_path)

    # Handle potential mismatch due to debug subsampling
    if len(test_ids) != len(final_preds):
        print(
            f"Adjusting Test IDs length from {len(test_ids)} to {len(final_preds)} to match predictions."
        )
        test_ids = test_ids[: len(final_preds)]

    # Create submission DataFrame
    submission = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
