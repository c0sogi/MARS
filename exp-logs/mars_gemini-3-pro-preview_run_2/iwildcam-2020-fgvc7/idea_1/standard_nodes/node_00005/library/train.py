import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import CameraTrapDataset
from library.model import EfficientNetClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: torch.device.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
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

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: torch.device.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def generate_predictions(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader for test data.
        device: torch.device.
        output_path: Path to save the submission CSV.
    """
    model.eval()
    ids = []
    predictions = []

    print("Starting inference on test set...")
    with torch.no_grad():
        for images, img_ids in loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)

            ids.extend(img_ids)
            predictions.extend(predicted_indices.cpu().numpy())

    # Create DataFrame
    df_submission = pd.DataFrame({"Id": ids, "Category": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    sample_size=None,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=3,
    load_cached_data=True,
):
    """
    Orchestrates the training process with early stopping.

    Args:
        sample_size (int, optional): Limit dataset size for debugging.
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        patience (int): Early stopping patience.
        load_cached_data (bool): Whether to use cached MegaDetector results.

    Returns:
        model: The trained model (best state).
    """
    Config.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Prepare Datasets and Loaders
    train_dataset = CameraTrapDataset(
        split="train", sample_size=sample_size, load_cached_data=load_cached_data
    )
    val_dataset = CameraTrapDataset(
        split="val", sample_size=sample_size, load_cached_data=load_cached_data
    )

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

    # 2. Initialize Model, Loss, Optimizer
    # Cite solution_lesson_node_00004: Modernize Backbones
    model = EfficientNetClassifier(num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # Cite solution_lesson_node_00004: Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Cite solution_lesson_node_00004: AdamW
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Cite solution_lesson_node_00004: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop with Early Stopping
    best_val_acc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model_custom.pth")
    Config.make_dirs()

    print("Starting training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc: {val_acc}")
        print(f"LR: {scheduler.get_last_lr()[0]}")

        # Checkpoint and Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("New best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 4. Load best model
    print(f"Loading best model with Val Acc: {best_val_acc}")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model
