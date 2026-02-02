import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_device
from library.features import FeaturePipeline
from library.dataset import ArenaDataset


class ClassifierMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) classifier for the Chatbot Arena task.
    Architecture: Input -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> Output
    """

    def __init__(
        self,
        input_dim: int = Config.INPUT_DIM,
        hidden_dim: int = Config.HIDDEN_DIM,
        output_dim: int = Config.OUTPUT_DIM,
        dropout_rate: float = Config.DROPOUT_RATE,
    ):
        super(ClassifierMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Logits of shape (batch_size, output_dim).
        """
        return self.network(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            batch_size = features.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    return running_loss / dataset_size


def generate_predictions(model, dataloader, device):
    """
    Generates probability predictions for the test set.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def run_training_pipeline():
    """
    Main function to execute the training, validation, and submission generation pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading & Processing
    # FeaturePipeline handles caching internally
    pipeline = FeaturePipeline()
    X_train, y_train, X_val, y_val, X_test, test_ids = pipeline.process_data()

    # Create Datasets
    train_dataset = ArenaDataset(X_train, y_train)
    val_dataset = ArenaDataset(X_val, y_val)
    test_dataset = ArenaDataset(X_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 3. Model Initialization
    model = ClassifierMLP().to(device)

    # Loss and Optimizer
    # CrossEntropyLoss supports soft labels (probabilities) directly
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # 5. Inference and Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    print("Generating predictions on test set...")
    predictions = generate_predictions(model, test_loader, device)

    # Format Submission
    submission_df = pd.DataFrame(
        predictions, columns=["winner_model_a", "winner_model_b", "winner_tie"]
    )
    submission_df.insert(0, "id", test_ids)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
