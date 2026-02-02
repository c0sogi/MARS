import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from library.utils import save_checkpoint


def train_one_epoch(
    model, dataloader, optimizer, device, epoch, genus_weight=0.5, print_freq=100
):
    """
    Trains the model for one epoch using the hierarchical loss.

    Args:
        model: The PyTorch model (HierarchicalResNet).
        dataloader: Training DataLoader.
        optimizer: Optimizer.
        device: Torch device.
        epoch: Current epoch number.
        genus_weight: Weight for the auxiliary genus loss.
        print_freq: Frequency of printing batch status (suppressed per requirements).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    criterion = nn.CrossEntropyLoss()

    for i, (images, species_ids, genus_ids) in enumerate(dataloader):
        images = images.to(device)
        species_ids = species_ids.to(device)
        genus_ids = genus_ids.to(device)

        # Forward pass
        # Pass species_ids to enable ArcFace margin
        species_logits, genus_logits = model(images, species_label=species_ids)

        # Calculate losses
        loss_species = criterion(species_logits, species_ids)
        loss_genus = criterion(genus_logits, genus_ids)

        total_loss = loss_species + (genus_weight * loss_genus)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update metrics
        batch_size = images.size(0)
        running_loss += total_loss.item() * batch_size
        count += batch_size

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        device: Torch device.

    Returns:
        float: Macro F1 score.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, species_ids, _ in dataloader:
            images = images.to(device)

            # Forward pass in inference mode (no labels passed)
            # This returns scaled cosine similarities without margin
            species_logits, _ = model(images, species_label=None)

            # Get predictions
            preds = torch.argmax(species_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(species_ids.numpy())

    # Calculate Macro F1
    # Using zero_division=0 to handle cases where a class is not predicted
    score = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return score


def generate_submission(model, dataloader, device, output_dir="./submission"):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model: The trained PyTorch model.
        dataloader: Test DataLoader.
        device: Torch device.
        output_dir: Directory to save the submission file.
    """
    model.eval()

    ids = []
    predictions = []

    with torch.no_grad():
        for images, image_ids in dataloader:
            images = images.to(device)

            # Forward pass in inference mode
            species_logits, _ = model(images, species_label=None)

            preds = torch.argmax(species_logits, dim=1)

            ids.extend(image_ids.numpy())
            predictions.extend(preds.cpu().numpy())

    # Create DataFrame
    df = pd.DataFrame({"Id": ids, "Predicted": predictions})

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")

    # Save to CSV
    df.to_csv(submission_path, index=False)
    # print(f"Submission saved to {submission_path}")


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    scheduler=None,
    genus_weight=0.5,
    patience=5,
    checkpoint_dir="./working/idea_4/",
):
    """
    Main training loop with early stopping and checkpointing.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        device: Torch device.
        num_epochs: Maximum number of epochs.
        scheduler: Learning rate scheduler (optional).
        genus_weight: Weight for auxiliary loss.
        patience: Early stopping patience.
        checkpoint_dir: Directory to save checkpoints.

    Returns:
        model: The model with the best weights loaded.
    """
    best_f1 = 0.0
    epochs_no_improve = 0

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, genus_weight=genus_weight
        )

        # Validate
        val_f1 = validate(model, val_loader, device)

        # Step scheduler
        if scheduler:
            scheduler.step()

        # Print metrics (Full precision)
        print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Macro F1 = {val_f1}")

        # Checkpoint and Early Stopping
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_f1": best_f1,
            },
            is_best,
            checkpoint_dir,
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # Load best model before returning
    best_path = os.path.join(checkpoint_dir, "model_best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)

        # Handle DataParallel wrapping if necessary
        state_dict = checkpoint["state_dict"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        # print(f"Loaded best model with F1: {checkpoint['best_f1']}")

    return model
