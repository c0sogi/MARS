import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.dataset import get_dataloaders


def build_model():
    """
    Constructs a ResNet-50 model with a binary classification head.
    Loads ImageNet weights and replaces the final fully connected layer.
    """
    # Load pre-trained ResNet-50 with default ImageNet weights
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)

    # Replace the final fully connected layer
    # Original: Linear(in_features=2048, out_features=1000)
    in_features = model.fc.in_features

    # New head for binary classification (1 output logit)
    model.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    # Move model to the configured device
    model = model.to(Config.DEVICE)

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        # Reshape labels to match model output (Batch, 1)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / total_samples


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return running_loss / total_samples


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            outputs = model(images)
            # Apply Sigmoid to logits to get probabilities [0, 1]
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_ids.extend(ids.numpy())
            all_probs.extend(probs)

    return all_ids, all_probs


def run_training(epochs=Config.EPOCHS, lr=Config.LEARNING_RATE, debug=False):
    """
    Main pipeline: Data loading, Model building, Training, Validation, and Submission.

    Args:
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        debug (bool): If True, uses a small subset of data.
    """
    print(f"Starting training pipeline (Epochs: {epochs}, LR: {lr}, Debug: {debug})...")

    # 1. Load Data
    dataloaders = get_dataloaders(debug=debug)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # 2. Build Model
    model = build_model()

    # 3. Setup Optimization
    # BCEWithLogitsLoss includes Sigmoid + BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop with Early Stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss = validate(model, val_loader, criterion, Config.DEVICE)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Best Validation Loss: {best_val_loss:.10f}")

    # 5. Generate Submission
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("Generating submission predictions...")
    ids, probs = predict(model, test_loader, Config.DEVICE)

    submission_df = pd.DataFrame({"id": ids, "label": probs})

    # Ensure IDs are integers and sorted
    submission_df["id"] = submission_df["id"].astype(int)
    submission_df = submission_df.sort_values("id")

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
