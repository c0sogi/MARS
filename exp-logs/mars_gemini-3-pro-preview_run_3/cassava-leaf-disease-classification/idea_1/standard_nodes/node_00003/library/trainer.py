import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataset
from library.model import CassavaResNet


def calculate_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Calculates class weights for Weighted Cross Entropy Loss.
    Weight = Total_Samples / (Num_Classes * Class_Count)
    """
    label_counts = df["label"].value_counts().sort_index()
    total_samples = len(df)
    num_classes = len(label_counts)

    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 0)
        if count > 0:
            weight = total_samples / (num_classes * count)
        else:
            weight = 1.0  # Fallback for missing classes in debug subsets
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(loader, model, criterion, optimizer, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds / total_preds

    return epoch_loss, epoch_acc


def validate(loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds / total_preds

    return epoch_loss, epoch_acc


def run_training(
    debug: bool = Config.DEBUG,
    epochs: int = Config.NUM_EPOCHS,
    patience: int = Config.PATIENCE,
    batch_size: int = Config.BATCH_SIZE,
    learning_rate: float = Config.LEARNING_RATE,
):
    """
    Main function to execute the training pipeline with Early Stopping.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Setup directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading datasets...")
    train_dataset = get_dataset("train", debug=debug)
    val_dataset = get_dataset("val", debug=debug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model Setup ---
    print("Initializing model...")
    model = CassavaResNet(pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # --- Loss Function (Weighted) ---
    # We need the dataframe to calculate weights. Accessing it from the dataset object.
    class_weights = calculate_class_weights(train_dataset.df, device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    best_val_acc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            train_loader, model, criterion, optimizer, device
        )
        val_loss, val_acc = validate(val_loader, model, criterion, device)

        # Print full precision metrics
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc: {val_acc}")

        # Early Stopping and Checkpointing
        if val_acc > best_val_acc:
            print(
                f"Validation accuracy improved from {best_val_acc} to {val_acc}. Saving model..."
            )
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement in validation accuracy. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc}")
    print(f"Best model saved to: {Config.MODEL_CHECKPOINT_PATH}")
