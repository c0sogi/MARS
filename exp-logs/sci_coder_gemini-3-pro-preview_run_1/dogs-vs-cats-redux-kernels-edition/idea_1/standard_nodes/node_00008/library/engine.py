import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from library.utils import Config, set_seed
from library.model import EfficientNetV2B2
from library.dataset import create_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Trains the model for one epoch using Mixed Precision.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: Device to train on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Match shape (Batch, 1)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using TTA (Horizontal Flip).

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            with autocast():
                # TTA: Original + Horizontal Flip
                logits = model(images)
                logits_flipped = model(torch.flip(images, dims=[3]))

                # Average probabilities
                probs = (torch.sigmoid(logits) + torch.sigmoid(logits_flipped)) / 2.0

                # Calculate loss on averaged probabilities
                loss = nn.BCELoss()(probs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def train_model(config: Config):
    """
    Main training loop with early stopping and model saving.

    Args:
        config (Config): Configuration object containing hyperparameters and paths.

    Returns:
        model: The trained model with the best weights loaded.
    """
    set_seed(config.seed)
    device = config.device

    # Create DataLoaders
    loaders = create_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Initialize model
    # Cite solution_lesson_node_00003: Use B2 model for increased capacity
    model = EfficientNetV2B2(pretrained=True)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    # Cite solution_lesson_node_00003: Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    scaler = GradScaler()

    # Early Stopping parameters
    best_val_loss = float("inf")
    patience = 2
    counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), config.model_path)
            # print(f"Model saved to {config.model_path}")
        else:
            counter += 1
            # print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss}")

    # Load best weights before returning
    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path, map_location=device))

    return model


def predict_and_submit(config: Config):
    """
    Loads the best model, performs inference on the test set, and saves the submission file.

    Args:
        config (Config): Configuration object.
    """
    device = config.device

    # Create DataLoaders (we only need test here)
    loaders = create_dataloaders(config)
    test_loader = loaders["test"]

    # Load model structure
    model = EfficientNetV2B2(pretrained=False)  # Weights will be loaded

    # Load best weights
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(
            f"Model file not found at {config.model_path}. Train model first."
        )

    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    ids = []
    probs = []

    print("Generating predictions...")

    with torch.no_grad():
        for images, batch_ids in test_loader:
            images = images.to(device)

            # Inference with TTA
            with autocast():
                logits = model(images)
                logits_flipped = model(torch.flip(images, dims=[3]))
                preds = (torch.sigmoid(logits) + torch.sigmoid(logits_flipped)) / 2.0

            # Store results
            ids.extend(batch_ids.numpy())
            probs.extend(preds.cpu().numpy().flatten())

    # Create submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "label": probs})

    # Sort by ID to ensure consistency
    df_sub = df_sub.sort_values("id")

    # Save to CSV
    # Ensure directory exists
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)
    df_sub.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
