import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_dataloaders
from library.model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        # Ensure labels match model output shape (Batch, 1)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / total_samples


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using TTA. Cite {solution_lesson_node_00007}
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    # Use BCELoss for averaged probabilities
    loss_fn = nn.BCELoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            # TTA: Original + Horizontal Flip
            outputs1 = model(images)
            outputs2 = model(torch.flip(images, dims=[3]))

            # Average probabilities
            probs1 = torch.sigmoid(outputs1)
            probs2 = torch.sigmoid(outputs2)
            avg_probs = (probs1 + probs2) / 2.0

            # Clamp for numerical stability
            avg_probs = torch.clamp(avg_probs, 1e-7, 1 - 1e-7)

            loss = loss_fn(avg_probs, labels)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return running_loss / total_samples


def predict(model, loader, device):
    """
    Generates predictions for the test set using TTA. Cite {solution_lesson_node_00007}
    """
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # TTA: Original + Horizontal Flip
            outputs1 = model(images)
            outputs2 = model(torch.flip(images, dims=[3]))

            # Average probabilities
            probs1 = torch.sigmoid(outputs1)
            probs2 = torch.sigmoid(outputs2)
            avg_probs = (probs1 + probs2) / 2.0

            probs = avg_probs.cpu().numpy().flatten()

            all_ids.extend(ids.numpy())
            all_probs.extend(probs)

    return all_ids, all_probs


def run_training(epochs=Config.EPOCHS, lr=Config.LEARNING_RATE, debug=False):
    """
    Main pipeline: Data loading, Model building, Training, Validation, and Submission.
    """
    print(f"Starting training pipeline (Epochs: {epochs}, LR: {lr}, Debug: {debug})...")

    # 1. Load Data
    dataloaders = get_dataloaders(debug=debug)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # 2. Build Model
    model = build_model()
    device = Config.DEVICE

    # 3. Setup Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop with Early Stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
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

    print(f"Best Validation Loss: {best_val_loss}")

    # 5. Generate Submission
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("Generating submission predictions...")
    ids, probs = predict(model, test_loader, device)

    submission_df = pd.DataFrame({"id": ids, "label": probs})

    # Ensure IDs are integers and sorted
    submission_df["id"] = submission_df["id"].astype(int)
    submission_df = submission_df.sort_values("id")

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
