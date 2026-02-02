import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import save_checkpoint, load_checkpoint


class ShallowCNN(nn.Module):
    """
    A shallow Convolutional Neural Network for image classification.
    Structure: 3 Convolutional Blocks -> Flatten -> Fully Connected Head.
    """

    def __init__(self):
        super(ShallowCNN, self).__init__()

        # Block 1: Input (3, 32, 32) -> Output (32, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=Config.CHANNELS, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: Input (32, 16, 16) -> Output (64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: Input (64, 8, 8) -> Output (128, 4, 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification Head
        # Flattened size: 128 channels * 4 height * 4 width = 2048
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 4 * 4, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns: Average Loss, ROC AUC Score
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to logits to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate ROC AUC
    # Handle potential edge cases where a batch might only have one class
    try:
        auc_score = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_score = 0.5

    return epoch_loss, auc_score


def train_model(model, train_loader, val_loader, config=Config):
    """
    Main training loop with Early Stopping.
    """
    device = torch.device(config.DEVICE)
    model = model.to(device)

    # Binary Cross Entropy with Logits is more numerically stable
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc + config.EARLY_STOPPING_MIN_DELTA:
            best_auc = val_auc
            patience_counter = 0

            # Save the best model state
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                },
                config.MODEL_SAVE_PATH,
            )

        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Load the best model weights before returning
    if os.path.exists(config.MODEL_SAVE_PATH):
        load_checkpoint(config.MODEL_SAVE_PATH, model)

    return model


def predict_and_submit(model, test_loader, output_path, device):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    model = model.to(device)

    probs = []

    # Ensure no gradients are calculated
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Apply sigmoid to convert logits to probabilities
            batch_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            probs.extend(batch_probs)

    # Retrieve IDs from the dataset metadata
    # We assume the test_loader is sequential (shuffle=False)
    test_metadata = test_loader.dataset.metadata
    ids = test_metadata["id"].values

    if len(ids) != len(probs):
        raise ValueError(
            f"Mismatch between number of IDs ({len(ids)}) and predictions ({len(probs)})"
        )

    # Create submission DataFrame
    df = pd.DataFrame({"id": ids, "has_cactus": probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
