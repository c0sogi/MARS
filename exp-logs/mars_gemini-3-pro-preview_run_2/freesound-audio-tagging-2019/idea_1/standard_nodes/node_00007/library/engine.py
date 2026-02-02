import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_lwlrap
from library.dataset import get_classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # Mixup Augmentation (Cite solution_lesson_node_00006)
        alpha = 1.0
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(images.size(0)).to(device)
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_targets = lam * targets + (1 - lam) * targets[index]

        optimizer.zero_grad()
        outputs = model(mixed_images)

        # Calculate loss with mixed targets
        loss = criterion(outputs, mixed_targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (Average validation loss, LWLRAP score)
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities for metric calculation
            preds = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    score = calculate_lwlrap(all_targets, all_preds)

    return losses.avg, score


def predict(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run inference on.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()
    all_preds = []
    all_fnames = []

    with torch.no_grad():
        for images, _, fnames in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_fnames.extend(fnames)

    all_preds = np.concatenate(all_preds, axis=0)

    # Retrieve class names to ensure correct column order
    classes = get_classes()

    # Create DataFrame
    df = pd.DataFrame(all_preds, columns=classes)
    df.insert(0, "fname", all_fnames)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run(model, train_loader, val_loader, test_loader, epochs, device, patience=7):
    """
    Main training loop with Early Stopping and Submission Generation.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        epochs (int): Maximum number of epochs.
        device (torch.device): Device to run on.
        patience (int): Early stopping patience.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler: Reduce LR when validation metric plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=2
    )

    best_score = 0.0
    patience_counter = 0
    best_model_path = Config.BEST_MODEL_PATH

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Step scheduler based on LWLRAP
        scheduler.step(val_score)

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val LWLRAP: {val_score}"
        )

        # Early Stopping and Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered. No improvement for {patience} epochs."
                )
                break

    print(f"Training finished. Best Validation LWLRAP: {best_score}")

    # Load best model for inference
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    # Generate Submission if threshold met
    if best_score > 0.7983677688266535:
        print(
            f"Validation score {best_score} exceeds threshold. Generating submission..."
        )
        predict(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation score {best_score} did not exceed threshold. Skipping submission."
        )
