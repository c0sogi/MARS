import os
import time
import torch
import numpy as np
from sklearn.metrics import f1_score
import random


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device, epoch_idx):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The hierarchical model.
        dataloader (DataLoader): Training data loader.
        criterion (HierarchicalLoss): The composite loss function.
        optimizer (Optimizer): PyTorch optimizer.
        device (torch.device): Device to train on.
        epoch_idx (int): Current epoch index (for logging).

    Returns:
        dict: Average losses for the epoch (total, species, genus, family).
    """
    model.train()

    running_loss_total = 0.0
    running_loss_species = 0.0
    running_loss_genus = 0.0
    running_loss_family = 0.0

    num_batches = len(dataloader)

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        # Move targets to device
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss, metrics = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss_total += metrics["loss_total"]
        running_loss_species += metrics["loss_species"]
        running_loss_genus += metrics["loss_genus"]
        running_loss_family += metrics["loss_family"]

    # Calculate averages
    avg_metrics = {
        "loss_total": running_loss_total / num_batches,
        "loss_species": running_loss_species / num_batches,
        "loss_genus": running_loss_genus / num_batches,
        "loss_family": running_loss_family / num_batches,
    }

    return avg_metrics


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The hierarchical model.
        dataloader (DataLoader): Validation data loader.
        criterion (HierarchicalLoss): The composite loss function.
        device (torch.device): Device to evaluate on.

    Returns:
        tuple: (avg_metrics, macro_f1_species)
    """
    model.eval()

    running_loss_total = 0.0
    running_loss_species = 0.0
    running_loss_genus = 0.0
    running_loss_family = 0.0

    all_preds = []
    all_targets = []

    num_batches = len(dataloader)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images)

            loss, metrics = criterion(outputs, targets)

            running_loss_total += metrics["loss_total"]
            running_loss_species += metrics["loss_species"]
            running_loss_genus += metrics["loss_genus"]
            running_loss_family += metrics["loss_family"]

            # Get predictions for species head (for F1 score)
            # outputs['species'] shape: (Batch, Num_Species)
            _, preds = torch.max(outputs["species"], 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets["species"].cpu().numpy())

    avg_metrics = {
        "loss_total": running_loss_total / num_batches,
        "loss_species": running_loss_species / num_batches,
        "loss_genus": running_loss_genus / num_batches,
        "loss_family": running_loss_family / num_batches,
    }

    # Calculate Macro F1 Score for Species
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_metrics, macro_f1


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    save_dir,
    patience=5,
):
    """
    Main training loop with Early Stopping and Checkpointing.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        criterion (HierarchicalLoss): Loss function.
        optimizer (Optimizer): Optimizer.
        num_epochs (int): Maximum number of epochs.
        device (torch.device): Device.
        save_dir (str): Directory to save checkpoints.
        patience (int): Early stopping patience.

    Returns:
        model: The model with the best weights loaded.
        dict: History of metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    best_f1 = -1.0
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics, val_f1 = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        # Store history
        history["train_loss"].append(train_metrics["loss_total"])
        history["val_loss"].append(val_metrics["loss_total"])
        history["val_f1"].append(val_f1)

        # Print metrics with full precision
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s")
        print(
            f"  Train Loss: Total={train_metrics['loss_total']}, Species={train_metrics['loss_species']}, Genus={train_metrics['loss_genus']}, Family={train_metrics['loss_family']}"
        )
        print(
            f"  Val Loss:   Total={val_metrics['loss_total']}, Species={val_metrics['loss_species']}"
        )
        print(f"  Val F1 (Macro): {val_f1}")

        # Checkpoint and Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  Validation F1 improved. Saved model to {best_model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(
                f"  Validation F1 did not improve. Patience: {epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation F1: {best_f1}")

    # Load best weights
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model, history
