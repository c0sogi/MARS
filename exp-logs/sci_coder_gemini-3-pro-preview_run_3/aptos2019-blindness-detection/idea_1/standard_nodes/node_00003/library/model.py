import os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from library.config import Config
from library.utils import compute_qwk


class ResNet18Regression(nn.Module):
    """
    ResNet18 architecture adapted for regression.
    Replaces the classification head with a single linear neuron.
    """

    def __init__(self, pretrained=True):
        super(ResNet18Regression, self).__init__()

        # Load pre-trained ResNet18 backbone
        # Using 'IMAGENET1K_V1' weights for transfer learning
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        # ResNet18's fc layer has 512 input features
        in_features = self.backbone.fc.in_features

        # Output is a single scalar for regression (severity score)
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        # Forward pass through the network
        return self.backbone(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        # Flatten outputs to match label shape (batch_size,)
        outputs = outputs.view(-1)

        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the Quadratic Weighted Kappa score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store predictions and labels for QWK calculation
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Calculate QWK metric using the utility function
    qwk_score = compute_qwk(all_labels, all_preds)

    return epoch_loss, qwk_score


def fit(model, train_loader, val_loader, config=Config):
    """
    Main training loop with Early Stopping.
    Saves the best model to config.MODEL_PATH.
    """
    device = config.DEVICE
    model.to(device)

    # Regression Loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    patience_counter = 0

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    print(f"Starting training on {device} for {config.NUM_EPOCHS} epochs.")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_qwk = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val QWK: {val_qwk:.4f}"
        )

        # Early Stopping Logic
        # We monitor Validation Loss (MSE) because it is the differentiable objective
        # and generally smoother than the QWK metric.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), config.MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # Load the best weights before returning
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        print("Loaded best model weights.")

    return model


def predict(model, test_loader, device=Config.DEVICE):
    """
    Generates predictions for the test set.
    Returns integer class labels after clipping and rounding.
    """
    model.eval()
    model.to(device)
    all_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(-1)
            all_preds.extend(outputs.cpu().numpy())

    # Post-processing for regression output
    preds_np = np.array(all_preds)

    # Clip to valid range [0, 4]
    preds_clipped = np.clip(preds_np, 0, 4)

    # Round to nearest integer for classification
    preds_rounded = np.round(preds_clipped).astype(int)

    return preds_rounded
