import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.utils import calculate_f1, set_seed


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion_species,
    criterion_aux,
    device,
    epoch,
    loss_weights=(1.0, 0.5, 0.5),
    use_amp=True,
):
    """
    Trains the model for one epoch using mixed precision.

    Args:
        model: The neural network model.
        dataloader: Training data loader.
        optimizer: Optimizer instance.
        criterion_species: Loss function for species (Focal Loss).
        criterion_aux: Loss function for auxiliary heads (Cross Entropy).
        device: Device to run on.
        epoch: Current epoch number.
        loss_weights: Tuple of weights for (species, genus, family) losses.
        use_amp: Boolean to enable Automatic Mixed Precision.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    w_s, w_g, w_f = loss_weights

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        # targets: (species, genus, family)
        species_label, genus_label, family_label = [
            t.to(device, non_blocking=True) for t in targets
        ]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            # Forward pass: returns (species_logits, genus_logits, fam_logits)
            species_logits, genus_logits, fam_logits = model(images)

            loss_s = criterion_species(species_logits, species_label)
            loss_g = criterion_aux(genus_logits, genus_label)
            loss_f = criterion_aux(fam_logits, family_label)

            loss = w_s * loss_s + w_g * loss_g + w_f * loss_f

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch} Training Loss: {avg_loss}")
    return avg_loss


def validate(
    model,
    dataloader,
    criterion_species,
    criterion_aux,
    device,
    loss_weights=(1.0, 0.5, 0.5),
):
    """
    Validates the model on the validation set.

    Returns:
        tuple: (average_loss, macro_f1_score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    w_s, w_g, w_f = loss_weights

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            species_label, genus_label, family_label = [
                t.to(device, non_blocking=True) for t in targets
            ]

            species_logits, genus_logits, fam_logits = model(images)

            loss_s = criterion_species(species_logits, species_label)
            loss_g = criterion_aux(genus_logits, genus_label)
            loss_f = criterion_aux(fam_logits, family_label)

            loss = w_s * loss_s + w_g * loss_g + w_f * loss_f
            running_loss += loss.item()

            # Predictions for species (target)
            preds = torch.argmax(species_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(species_label.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    f1 = calculate_f1(all_labels, all_preds)

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation Macro F1: {f1}")

    return avg_loss, f1


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion_species,
    criterion_aux,
    device,
    num_epochs,
    save_path,
    patience=5,
    loss_weights=(1.0, 0.5, 0.5),
    use_amp=True,
    seed=42,
):
    """
    Orchestrates the training process with early stopping.
    """
    set_seed(seed)

    best_f1 = -1.0
    epochs_no_improve = 0

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion_species,
            criterion_aux,
            device,
            epoch,
            loss_weights,
            use_amp,
        )

        val_loss, val_f1 = validate(
            model, val_loader, criterion_species, criterion_aux, device, loss_weights
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()

        # Early Stopping and Checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with F1: {best_f1}")
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best F1: {best_f1}")

    # Load best model for return
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def predict_test_set(
    model, test_loader, device, output_path="./submission/submission.csv"
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    ids = []
    predictions = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device, non_blocking=True)

            # Forward pass
            # Model returns (species_logits, genus_logits, fam_logits)
            species_logits, _, _ = model(images)

            preds = torch.argmax(species_logits, dim=1)

            ids.extend(image_ids.numpy())
            predictions.extend(preds.cpu().numpy())

    # Create DataFrame
    df = pd.DataFrame({"Id": ids, "Predicted": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
