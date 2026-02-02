import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    NUM_EPOCHS,
    BATCH_SIZE,
)
from library.utils import save_model, load_model
from library.network import IcebergResNet


def train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE):
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
    # Cite solution_lesson_node_00005: Use BCEWithLogitsLoss for numerical stability with logits
    criterion = nn.BCEWithLogitsLoss()
    # Cite solution_lesson_node_00005: Switch to AdamW
    optimizer = optim.AdamW(
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

            # Cite solution_lesson_node_00005: Label Smoothing (epsilon=0.05)
            # Soft targets: y_smooth = y * (1 - eps) + 0.5 * eps
            eps = 0.05
            labels_smooth = labels * (1.0 - eps) + 0.5 * eps

            loss = criterion(outputs, labels_smooth)

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
                # Validation loss on hard targets for metric tracking
                # We use BCEWithLogitsLoss here against true labels
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)

                # Accuracy calculation (Threshold 0.5 on probabilities)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
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
    if os.path.exists(MODEL_SAVE_PATH):
        model = load_model(model, MODEL_SAVE_PATH, device=DEVICE)
    else:
        print("Warning: No model file found to load. Returning current model.")

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

            # Cite solution_lesson_node_00005: Test Time Augmentation (TTA)
            # 1. Original
            out1 = model(images, angles)
            prob1 = torch.sigmoid(out1)

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            out2 = model(images_h, angles)
            prob2 = torch.sigmoid(out2)

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[2])
            out3 = model(images_v, angles)
            prob3 = torch.sigmoid(out3)

            # Average
            avg_prob = (prob1 + prob2 + prob3) / 3.0

            preds = avg_prob.cpu().numpy().flatten()
            predictions.extend(preds)

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
