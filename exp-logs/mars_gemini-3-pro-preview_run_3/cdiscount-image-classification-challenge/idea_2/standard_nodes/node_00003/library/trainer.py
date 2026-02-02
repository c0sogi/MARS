import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader

from library.config import (
    DEVICE,
    MLP_BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    MODEL_SAVE_PATH,
    NUM_WORKERS,
    EMBEDDING_DIM,
    NUM_CLASSES,
)
from library.models import ProductClassifier
from library.data_loader import EmbeddingDataset


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the provided loader.
    Returns average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def train_mlp(
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    input_dim=EMBEDDING_DIM,
    num_classes=NUM_CLASSES,
    batch_size=MLP_BATCH_SIZE,
    lr=LEARNING_RATE,
    epochs=EPOCHS,
    patience=PATIENCE,
    device=DEVICE,
    save_path=MODEL_SAVE_PATH,
):
    """
    Trains the MLP classifier on pre-computed embeddings.
    Implements Early Stopping and saves the best model.
    """
    # Prepare Datasets and Loaders
    # We use num_workers=0 for EmbeddingDataset usually as it is in-memory,
    # but we stick to config if needed. For simple tensor data, 0 is often faster/safer.
    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    val_dataset = EmbeddingDataset(val_embeddings, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    # Initialize Model
    model = ProductClassifier(input_dim=input_dim, num_classes=num_classes)
    model.to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training State
    best_acc = -1.0
    patience_counter = 0

    print(f"Starting MLP training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_train += inputs.size(0)

        train_loss = running_loss / total_train

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time} seconds.")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Accuracy: {val_acc}")

        # Early Stopping Check
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), save_path)
            print(f"Validation accuracy improved. Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best Validation Accuracy: {best_acc}")

    # Reload best model weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def predict_mlp(model, test_embeddings, batch_size=MLP_BATCH_SIZE, device=DEVICE):
    """
    Generates predictions for the test set using the trained model.
    """
    dataset = EmbeddingDataset(test_embeddings, labels=None)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    model.to(device)
    model.eval()

    all_predictions = []

    print(f"Starting prediction on {len(dataset)} samples...")

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.append(predicted.cpu().numpy())

    return np.concatenate(all_predictions)
