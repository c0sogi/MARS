import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_macro_f1, set_seed
from library.loss import FocalLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        loader: DataLoader for the training set.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to run training on.

    Returns:
        float: Average training loss.
    """
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for the validation set.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        tuple: (Average validation loss, Macro F1 score)
    """
    model.eval()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate metric
    f1 = calculate_macro_f1(all_targets, all_preds)

    return losses.avg, f1


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=Config.NUM_EPOCHS,
    patience=Config.PATIENCE,
):
    """
    Main training loop with early stopping.

    Args:
        model: The neural network model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Device to run on.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience.

    Returns:
        model: The model with the best weights loaded.
    """
    set_seed()

    # Initialize optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Use FocalLoss as specified
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA).to(device)

    best_f1 = -1.0
    patience_counter = 0
    best_model_path = Config.BEST_MODEL_PATH

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val F1 = {val_f1}"
        )

        # Early Stopping Logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training complete. Best Val F1: {best_f1}")

    # Load best weights before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model


def predict(
    model, test_loader, classes_list, device, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained neural network model.
        test_loader: DataLoader for the test set.
        classes_list: List mapping model indices to original category_ids.
        device: Device to run inference on.
        output_path: Path to save the submission CSV.
    """
    # Ensure model is in eval mode and load best weights if available
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    model.eval()

    ids_list = []
    predictions_list = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            ids_list.extend(image_ids.numpy())
            predictions_list.extend(preds)

    # Map predictions back to original category_ids
    # classes_list is sorted such that index i corresponds to the category_id at that index
    final_preds = [classes_list[p] for p in predictions_list]

    # Create DataFrame
    submission_df = pd.DataFrame({"Id": ids_list, "Predicted": final_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
