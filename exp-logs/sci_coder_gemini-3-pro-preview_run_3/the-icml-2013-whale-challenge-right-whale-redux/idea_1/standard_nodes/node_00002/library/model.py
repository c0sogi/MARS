import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.dataset import get_dataloaders


class ShallowCNN(nn.Module):
    """
    A lightweight 2D CNN for Right Whale Detection using Log-Mel Spectrograms.
    Architecture:
        - 3 Convolutional Blocks (Conv3x3 -> BN -> ReLU -> MaxPool2x2)
        - Global Average Pooling
        - Dense Output Layer with Sigmoid Activation
    """

    def __init__(self):
        super(ShallowCNN, self).__init__()

        # Block 1
        # Input: (B, 1, 64, 32) -> Output: (B, 32, 32, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        # Input: (B, 32, 32, 16) -> Output: (B, 64, 16, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3
        # Input: (B, 64, 16, 8) -> Output: (B, 128, 8, 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Head
        # Global Average Pooling collapses spatial dims (8, 4) -> (1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=Config.DROPOUT)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


def train_model():
    """
    Trains the ShallowCNN model with Early Stopping and Class Weighting.
    Returns:
        str: Path to the saved best model weights.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load Data
    print("Loading data...")
    dataloaders = get_dataloaders()
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Initialize Model
    model = ShallowCNN().to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Loss Function
    # Using BCELoss with manual weighting for class imbalance
    criterion = nn.BCELoss(reduction="none")

    # Training State
    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(
        f"Starting training for {Config.EPOCHS} epochs with patience {Config.PATIENCE}..."
    )

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)

            # Calculate weighted loss
            loss_unreduced = criterion(outputs, labels)
            # Apply weight: POS_WEIGHT for label 1, 1.0 for label 0
            weights = labels * Config.POS_WEIGHT + (1 - labels)
            loss = (loss_unreduced * weights).mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                outputs = model(inputs)

                # Validation loss
                loss_unreduced = criterion(outputs, labels)
                weights = labels * Config.POS_WEIGHT + (1 - labels)
                loss = (loss_unreduced * weights).mean()

                val_running_loss += loss.item() * inputs.size(0)

                # Store for AUC
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader.dataset)

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_auc = 0.5

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {epoch_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f} | "
            f"Val AUC: {val_auc:.6f}"
        )

        # --- Early Stopping Check ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_val_auc:.6f}"
                )
                break

    print(f"Training finished. Best model saved to {best_model_path}")
    return best_model_path


def generate_submission(model_path):
    """
    Loads the best model and generates predictions for the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating submission using model at {model_path}...")

    # Load Data
    dataloaders = get_dataloaders()
    test_loader = dataloaders["test"]

    # Load Model
    model = ShallowCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    clip_names_list = []

    with torch.no_grad():
        for inputs, clip_names in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)

            # Flatten predictions
            probs = outputs.cpu().numpy().flatten()

            predictions.extend(probs)
            clip_names_list.extend(clip_names)

    # Create Submission DataFrame
    df = pd.DataFrame({"clip": clip_names_list, "probability": predictions})

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    """
    Main execution pipeline.
    """
    best_model_path = train_model()
    generate_submission(best_model_path)
