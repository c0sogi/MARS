import torch
import time
import numpy as np
from sklearn.metrics import f1_score
from library.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SCHEDULER_T_MAX,
    SCHEDULER_MIN_LR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    BEST_MODEL_PATH,
)
from library.loss import HierarchicalLoss


def get_optimizer(model):
    """
    Creates the AdamW optimizer based on config parameters.
    """
    return torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )


def get_scheduler(optimizer):
    """
    Creates the Cosine Annealing Learning Rate Scheduler.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_T_MAX, eta_min=SCHEDULER_MIN_LR
    )


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Performs one epoch of training.

    Args:
        model: The HierarchicalEfficientNet model.
        dataloader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The HierarchicalLoss function.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch index.

    Returns:
        dict: Average loss metrics for the epoch.
    """
    model.train()

    running_loss_total = 0.0
    running_loss_species = 0.0
    running_loss_genus = 0.0
    running_loss_family = 0.0

    num_batches = len(dataloader)

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        # Unpack targets (list of tensors) and move to device
        # targets structure: [species_tensor, genus_tensor, family_tensor]
        species_targets = targets[0].to(device)
        genus_targets = targets[1].to(device)
        family_targets = targets[2].to(device)

        target_tuple = (species_targets, genus_targets, family_targets)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss, metrics = criterion(outputs, target_tuple)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss_total += metrics["loss_total"]
        running_loss_species += metrics["loss_species"]
        running_loss_genus += metrics["loss_genus"]
        running_loss_family += metrics["loss_family"]

    return {
        "loss_total": running_loss_total / num_batches if num_batches > 0 else 0.0,
        "loss_species": running_loss_species / num_batches if num_batches > 0 else 0.0,
        "loss_genus": running_loss_genus / num_batches if num_batches > 0 else 0.0,
        "loss_family": running_loss_family / num_batches if num_batches > 0 else 0.0,
    }


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Calculates losses and the Macro F1 score for species predictions.

    Args:
        model: The HierarchicalEfficientNet model.
        dataloader: DataLoader for validation data.
        criterion: The HierarchicalLoss function.
        device: 'cuda' or 'cpu'.

    Returns:
        dict: Average loss metrics and Macro F1 score.
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

            species_targets = targets[0].to(device)
            genus_targets = targets[1].to(device)
            family_targets = targets[2].to(device)

            target_tuple = (species_targets, genus_targets, family_targets)

            outputs = model(images)

            loss, metrics = criterion(outputs, target_tuple)

            running_loss_total += metrics["loss_total"]
            running_loss_species += metrics["loss_species"]
            running_loss_genus += metrics["loss_genus"]
            running_loss_family += metrics["loss_family"]

            # Get predictions for Species Head (Target)
            # outputs['species'] shape: (Batch, Num_Species)
            preds = torch.argmax(outputs["species"], dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(species_targets.cpu().numpy())

    # Calculate Macro F1 Score
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        macro_f1 = f1_score(all_targets, all_preds, average="macro")
    else:
        macro_f1 = 0.0

    return {
        "loss_total": running_loss_total / num_batches if num_batches > 0 else 0.0,
        "loss_species": running_loss_species / num_batches if num_batches > 0 else 0.0,
        "loss_genus": running_loss_genus / num_batches if num_batches > 0 else 0.0,
        "loss_family": running_loss_family / num_batches if num_batches > 0 else 0.0,
        "macro_f1": macro_f1,
    }


def train_model(model, train_loader, val_loader, num_epochs):
    """
    Main training loop with Early Stopping and Scheduler.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        num_epochs: Maximum number of epochs.

    Returns:
        model: The trained model (loaded with best weights).
    """
    device = DEVICE
    model.to(device)

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = HierarchicalLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train Phase
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validation Phase
        val_metrics = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        epoch_time = time.time() - start_time

        # Logging
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(
            f"Train Loss: {train_metrics['loss_total']} (Species: {train_metrics['loss_species']}, Genus: {train_metrics['loss_genus']}, Family: {train_metrics['loss_family']})"
        )
        print(
            f"Val Loss: {val_metrics['loss_total']} (Species: {val_metrics['loss_species']}, Genus: {val_metrics['loss_genus']}, Family: {val_metrics['loss_family']})"
        )
        print(f"Val Macro F1: {val_metrics['macro_f1']}")

        # Early Stopping Logic (Monitoring Total Validation Loss)
        current_val_loss = val_metrics["loss_total"]

        if current_val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = current_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved to {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(
                f"Early stopping counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")

    # Load best model weights before returning
    if best_val_loss != float("inf"):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))

    return model
