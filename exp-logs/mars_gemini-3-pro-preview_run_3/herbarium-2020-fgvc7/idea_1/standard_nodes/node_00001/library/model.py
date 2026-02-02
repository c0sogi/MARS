import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import library.config as config


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 based classifier for plant species.
    Replaces the final fully connected layer to match the number of classes.
    """

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        super(ResNet18Classifier, self).__init__()

        # Load ResNet18 backbone
        # Using weights parameter for newer torchvision versions
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        # ResNet18 fc input features is 512
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.
    Returns loss, accuracy, and macro F1 score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    total = len(dataloader.dataset)
    epoch_loss = running_loss / total

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    epoch_acc = (all_preds == all_labels).mean()

    # Calculate Macro F1
    # Using zero_division=0 to handle classes not present in the batch/split gracefully
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


def train_model(model, train_loader, val_loader):
    """
    Main training loop with early stopping.
    """
    device = torch.device(config.DEVICE)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Scheduler to reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_f1 = 0.0
    patience_counter = 0

    print(f"Starting training on device: {device}")
    print(f"Epochs: {config.NUM_EPOCHS}, Batch Size: {config.BATCH_SIZE}")

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"  Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f}")

        # Validate
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        print(
            f"  Val Loss:   {val_loss:.6f} | Val Acc:   {val_acc:.6f} | Val F1 (Macro): {val_f1:.6f}"
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping & Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  -> Model saved! (New best val_loss: {best_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{config.PATIENCE}"
            )

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(
        f"Training complete. Best Val Loss: {best_loss:.6f}, Best Val F1: {best_f1:.6f}"
    )

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model


def generate_submission(model, test_loader):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    device = torch.device(config.DEVICE)
    model = model.to(device)
    model.eval()

    print("Generating predictions for test set...")

    ids = []
    predictions = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            ids.extend(image_ids.numpy())
            predictions.extend(preds.cpu().numpy())

    # Create DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Predicted": predictions})

    # Ensure Id is integer
    submission_df["Id"] = submission_df["Id"].astype(int)

    # Sort by Id just in case
    submission_df = submission_df.sort_values(by="Id")

    # Save
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print(submission_df.head())
