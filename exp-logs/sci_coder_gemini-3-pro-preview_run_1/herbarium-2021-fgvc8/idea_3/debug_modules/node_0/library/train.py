import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, get_device, calculate_macro_f1, AverageMeter
from library.dataset import get_dataloaders
from library.model import HierarchicalEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, species_targets, family_targets) in enumerate(loader):
        images = images.to(device)
        species_targets = species_targets.to(device)
        family_targets = family_targets.to(device)

        optimizer.zero_grad()

        # Forward pass: returns (species_logits, family_logits)
        species_logits, family_logits = model(images)

        # Calculate losses for both tasks
        loss_species = criterion(species_logits, species_targets)
        loss_family = criterion(family_logits, family_targets)

        # Weighted sum of losses
        loss = loss_species + Config.FAMILY_LOSS_WEIGHT * loss_family

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, species_targets, family_targets in loader:
            images = images.to(device)
            species_targets = species_targets.to(device)
            family_targets = family_targets.to(device)

            species_logits, family_logits = model(images)

            loss_species = criterion(species_logits, species_targets)
            loss_family = criterion(family_logits, family_targets)

            loss = loss_species + Config.FAMILY_LOSS_WEIGHT * loss_family

            losses.update(loss.item(), images.size(0))

            # Get predictions for species (primary task)
            preds = torch.argmax(species_logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(species_targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Macro F1 Score
    f1 = calculate_macro_f1(all_targets, all_preds)

    return losses.avg, f1


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)

            # We only care about species logits for submission
            species_logits, _ = model(images)
            preds = torch.argmax(species_logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_ids.append(image_ids.numpy())

    all_preds = np.concatenate(all_preds)
    all_ids = np.concatenate(all_ids)

    df = pd.DataFrame({"Id": all_ids, "Predicted": all_preds})

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def fit(epochs=Config.NUM_EPOCHS, debug=Config.DEBUG):
    """
    Main training loop.
    """
    set_seed(Config.SEED)
    device = get_device()

    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader, num_families = get_dataloaders(debug=debug)

    # 2. Initialize Model
    model = HierarchicalEfficientNet(
        num_families=num_families,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
    )
    model = model.to(device)

    # 3. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Scheduler (OneCycleLR)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=100.0,
    )

    # 5. Training Loop
    best_f1 = -1.0
    patience = 3
    patience_counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}/{epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F1: {val_f1}")

        # Checkpointing & Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with F1: {best_f1}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")

    # 6. Generate Submission
    print("Generating submission...")
    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
