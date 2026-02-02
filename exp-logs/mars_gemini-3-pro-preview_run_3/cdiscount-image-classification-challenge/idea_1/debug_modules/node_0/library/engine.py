import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import DEVICE, SUBMISSION_PATH, WORKING_DIR, CACHE_DIR
from library.utils import calculate_class_weights, get_category_mapping


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_idx):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Compute device.
        epoch_idx (int): Current epoch index (0-based).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Aggregate loss (multiply by batch size to handle potential last partial batch)
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    print(f"Epoch {epoch_idx} Training Loss: {avg_loss}")

    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Compute device.

    Returns:
        float: Accuracy score (0.0 to 1.0).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    print(f"Validation Accuracy: {accuracy}")
    return accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    patience=3,
    load_cached_weights=True,
):
    """
    Full training pipeline with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Maximum number of epochs.
        device (torch.device): Compute device.
        patience (int): Early stopping patience.
        load_cached_weights (bool): Whether to use cached class weights.

    Returns:
        nn.Module: The best trained model state.
    """
    # Calculate and load class weights for imbalance handling
    class_weights = calculate_class_weights(load_cached_data=load_cached_weights)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # Optional: Learning rate scheduler could be added here
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=1
    )

    best_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_acc = evaluate(model, val_loader, device)

        scheduler.step(val_acc)

        # Checkpoint and Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_acc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model weights before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model weights.")

    return model


def generate_predictions(model, test_loader, device, output_path=SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves to CSV.
    Handles the variable number of images per product by averaging logits.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader (must have batch_size=1).
        device (torch.device): Compute device.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()

    # Get mapping from index back to category_id
    _, idx_to_id = get_category_mapping(load_cached_data=True)

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for img_stack, product_id in test_loader:
            # img_stack shape from loader (batch_size=1): (1, Num_Images, C, H, W)
            # product_id shape: (1,)

            # Remove batch dimension to get (Num_Images, C, H, W)
            # This allows us to treat the multiple views as a batch for the model
            images = img_stack.squeeze(0).to(device)

            # Forward pass
            logits = model(images)  # Shape: (Num_Images, Num_Classes)

            # Average logits across all images for this product
            avg_logits = torch.mean(logits, dim=0)  # Shape: (Num_Classes)

            # Get predicted class index
            pred_idx = torch.argmax(avg_logits).item()

            # Map to original category_id
            category_id = idx_to_id[pred_idx]

            results.append(
                {"_id": int(product_id.item()), "category_id": int(category_id)}
            )

    # Save to CSV
    df_submission = pd.DataFrame(results)
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
