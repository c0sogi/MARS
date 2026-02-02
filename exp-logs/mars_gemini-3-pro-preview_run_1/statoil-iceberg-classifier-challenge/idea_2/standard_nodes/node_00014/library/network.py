import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import pandas as pd
import numpy as np
import os

from library.config import (
    DEVICE,
    RESNET_FEATURE_DIM,
    ANGLE_DIM,
    DROPOUT_RATE,
    MODEL_SAVE_PATH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    SUBMISSION_PATH,
)
from library.utils import save_model, load_model


class IcebergResNet(nn.Module):
    """
    ResNet-18 based model with Late Fusion for Iceberg detection.

    Architecture:
    1. ResNet-18 Backbone (Pretrained on ImageNet)
    2. Global Average Pooling -> 512-dim vector
    3. Late Fusion: Concatenate 512-dim vector with 1-dim Incidence Angle
    4. Classification Head: Dense -> BN -> ReLU -> Dropout -> Sigmoid
    """

    def __init__(self):
        super(IcebergResNet, self).__init__()

        # Load pre-trained ResNet-18
        # Using default weights (ImageNet)
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # The input images are 3-channel (Band1, Band2, Mean), so we keep conv1 as is.
        # We remove the final fully connected layer.
        # Feature extractor includes everything up to the Global Average Pooling layer.
        # ResNet structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            self.backbone.avgpool,
        )

        # Late Fusion Head
        # Input: ResNet features (512) + Incidence Angle (1)
        input_dim = RESNET_FEATURE_DIM + ANGLE_DIM

        # We use an intermediate hidden dimension of 512 to match the visual feature size
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(512, 1),
        )

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, H, W)
            angle (torch.Tensor): Angle tensor of shape (B, 1)
        Returns:
            torch.Tensor: Probability of iceberg (B, 1)
        """
        # Extract visual features
        features = self.feature_extractor(x)  # Output: (B, 512, 1, 1)
        features = features.flatten(1)  # Output: (B, 512)

        # Concatenate with angle
        # angle is expected to be (B, 1)
        fused_features = torch.cat((features, angle), dim=1)  # Output: (B, 513)

        # Classification
        prob = self.classification_head(fused_features)  # Output: (B, 1)

        return prob


def train_model(train_loader, val_loader, num_epochs=30, patience=PATIENCE):
    """
    Trains the IcebergResNet model with Early Stopping and Scheduler.

    Args:
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
        num_epochs (int): Maximum number of epochs.
        patience (int): Patience for early stopping.

    Returns:
        nn.Module: The trained model with the best validation weights.
    """
    model = IcebergResNet().to(DEVICE)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Scheduler: Reduce LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for images, angles, labels in train_loader:
            images = images.to(DEVICE)
            angles = angles.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(DEVICE)
                angles = angles.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images, angles)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)

                # Accuracy calculation (Threshold 0.5)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = correct / total

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss} - Val Loss: {val_loss} - Val Acc: {val_acc}"
        )

        # --- Scheduler Step ---
        scheduler.step(val_loss)

        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, MODEL_SAVE_PATH)
            print(f"Validation loss improved. Model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model weights before returning
    print("Loading best model for return...")
    model = load_model(model, MODEL_SAVE_PATH, device=DEVICE)
    return model


def predict_and_submit(model, test_loader, test_ids, output_path=SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Loader for test data.
        test_ids (np.ndarray): Array of test image IDs.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(DEVICE)
            angles = angles.to(DEVICE)

            outputs = model(images, angles)
            # Outputs are already probabilities (Sigmoid)
            preds = outputs.cpu().numpy().flatten()
            predictions.extend(preds)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
