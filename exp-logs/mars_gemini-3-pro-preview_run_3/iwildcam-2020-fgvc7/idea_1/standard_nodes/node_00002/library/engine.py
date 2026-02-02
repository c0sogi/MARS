import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: 'cpu' or 'cuda'.

    Returns:
        avg_loss (float): Average training loss.
        avg_acc (float): Average training accuracy.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data).item()
        total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = correct_predictions / total_samples

    return avg_loss, avg_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: 'cpu' or 'cuda'.

    Returns:
        avg_loss (float): Average validation loss.
        avg_acc (float): Average validation accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data).item()
            total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = correct_predictions / total_samples

    return avg_loss, avg_acc


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=Config.EPOCHS,
    patience=Config.EARLY_STOPPING_PATIENCE,
    save_path=Config.MODEL_SAVE_PATH,
):
    """
    Runs the full training loop with early stopping.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        device: 'cpu' or 'cuda'.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.
    """
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.10f}, Train Acc: {train_acc:.10f}")
        print(f"Val Loss: {val_loss:.10f}, Val Acc: {val_acc:.10f}")

        # Early Stopping Logic
        # We save the model with the best accuracy as that is the competition metric.
        # However, we often monitor loss for early stopping convergence.
        # Here we will monitor validation loss for stopping, but save based on accuracy
        # (or save based on loss if accuracy is unstable, but accuracy is the goal).
        # Let's stick to monitoring Validation Loss for stability.

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print("Training complete.")


def generate_submission(
    model, test_loader, low_conf_df, device, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for high-confidence test images.
        low_conf_df: DataFrame containing low-confidence images (predicted as Empty).
        device: 'cpu' or 'cuda'.
        output_path: Path to save the CSV.
    """
    model.eval()
    predictions = []

    print("Generating predictions for high-confidence images...")
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds = preds.cpu().numpy()

            for img_id, pred_cat in zip(image_ids, preds):
                predictions.append({"Id": img_id, "Category": int(pred_cat)})

    print("Processing low-confidence images...")
    # Add low confidence images as class 0 (Empty)
    for img_id in low_conf_df["image_id"]:
        predictions.append({"Id": img_id, "Category": Config.EMPTY_CLASS_ID})

    # Create DataFrame
    submission_df = pd.DataFrame(predictions)

    # Sort by Id to ensure consistent order (optional but recommended)
    submission_df = submission_df.sort_values(by="Id")

    # Rename columns to match sample submission just in case, though we used correct names
    # Sample submission: Id, Category.
    # Wait, the task description says:
    # Submission Format:
    # Id,Predicted
    # 58857ccf...,1
    #
    # However, sample_submission.csv provided in dataset info shows:
    # Id, Category
    #
    # I will follow the "Submission Format" text in the description: "Id,Predicted".
    # But usually, it's safer to check sample submission.
    # The prompt text says: "The Id column corresponds to the test image id. The Category is an integer..."
    # But the text block explicitly shows:
    # Id,Predicted
    #
    # I will stick to "Id,Predicted" as requested in the specific Submission Format section.

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}. Total predictions: {len(submission_df)}")
