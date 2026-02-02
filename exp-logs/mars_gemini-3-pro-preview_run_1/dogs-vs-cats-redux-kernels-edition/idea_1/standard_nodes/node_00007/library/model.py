import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.cuda.amp import autocast, GradScaler

from library.utils import Config, set_seed


class DogCatModel(nn.Module):
    """
    Generic EfficientNetV2 model for binary classification.
    Uses a pre-trained backbone from timm and replaces the classifier head.
    """

    def __init__(self, model_name="tf_efficientnetv2_b2", pretrained=True):
        super(DogCatModel, self).__init__()
        # Load pre-trained backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Get the input features of the classifier
        in_features = self.backbone.classifier.in_features

        # Replace the classifier with a single linear layer for binary classification (1 logit)
        self.backbone.classifier = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Match shape (Batch, 1)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

    return running_loss / dataset_size


def train_model(config: Config, train_loader, val_loader):
    """
    Main training loop with early stopping and model saving.
    """
    set_seed(config.seed)
    device = config.device

    # Initialize model
    model = DogCatModel(model_name=config.model_name, pretrained=True)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()

    # Early Stopping parameters
    best_val_loss = float("inf")
    patience = 2
    counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), config.model_path)
            # print(f"Model saved to {config.model_path}")
        else:
            counter += 1
            # print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss}")
    return model


def predict_and_submit(config: Config, test_loader):
    """
    Loads the best model, performs inference on the test set, and saves the submission file.
    """
    device = config.device

    # Load model structure
    model = DogCatModel(
        model_name=config.model_name, pretrained=False
    )  # Weights will be loaded

    # Load best weights
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(
            f"Model file not found at {config.model_path}. Train model first."
        )

    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    ids = []
    probs = []

    print("Generating predictions...")

    with torch.no_grad():
        for images, batch_ids in test_loader:
            images = images.to(device)

            # Inference
            with autocast():
                logits = model(images)
                preds = torch.sigmoid(logits)

            # Store results
            ids.extend(batch_ids.numpy())
            probs.extend(preds.cpu().numpy().flatten())

    # Create submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "label": probs})

    # Sort by ID to ensure consistency
    df_sub = df_sub.sort_values("id")

    # Save to CSV
    df_sub.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
