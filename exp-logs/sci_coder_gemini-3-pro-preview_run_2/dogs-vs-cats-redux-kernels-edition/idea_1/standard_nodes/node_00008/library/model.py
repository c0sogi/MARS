import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models
from torchvision.models import ResNet34_Weights
from library.utils import get_device


class FineTunedResNet34(nn.Module):
    """
    A ResNet-34 model fine-tuned for binary classification.
    Replaces the final fully connected layer to output a single logit.
    """

    def __init__(self):
        super(FineTunedResNet34, self).__init__()
        # Load pre-trained ResNet-34
        # Cite solution_lesson_node_00004: Upgrade backbone to ResNet34
        self.backbone = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Replace the final FC layer
        # ResNet-34 fc layer has 512 input features
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.backbone(x)


def train_model(model, dataloaders, num_epochs=5, lr=1e-4, patience=3, device=None):
    """
    Trains the model using AdamW optimizer and BCEWithLogitsLoss.
    Implements Early Stopping based on validation loss.

    Args:
        model (nn.Module): The model to train.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        num_epochs (int): Maximum number of epochs.
        lr (float): Learning rate.
        patience (int): Early stopping patience.
        device (torch.device): Device to train on.

    Returns:
        model (nn.Module): The trained model with best weights loaded.
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_count = 0

        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)  # Match shape [Batch, 1]

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_count += inputs.size(0)

        epoch_train_loss = running_loss / train_count

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_count += inputs.size(0)

        epoch_val_loss = val_running_loss / val_count

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss} - Val Loss: {epoch_val_loss}"
        )

        # --- Early Stopping Logic ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model


def generate_submission(
    model, test_loader, output_path="./submission/submission.csv", device=None
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        output_path (str): Path to save the submission CSV.
        device (torch.device): Device to run inference on.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    ids_list = []
    probs_list = []

    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)
            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Flatten and move to CPU
            probs = probs.view(-1).cpu().numpy()
            ids = ids.cpu().numpy()

            ids_list.extend(ids)
            probs_list.extend(probs)

    # Create DataFrame
    submission_df = pd.DataFrame({"id": ids_list, "label": probs_list})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
