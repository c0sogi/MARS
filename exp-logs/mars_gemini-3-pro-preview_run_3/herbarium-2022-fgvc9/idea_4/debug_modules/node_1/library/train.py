import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_logger, AverageMeter, get_score
from library.dataset import get_loaders
from library.model import get_model


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, loss_weights
):
    """
    Trains the model for one epoch using hierarchical multi-task loss.
    """
    model.train()

    losses = AverageMeter()

    # Weights for the multi-task loss
    w_species = loss_weights.get("species", 1.0)
    w_genus = loss_weights.get("genus", 0.1)
    w_family = loss_weights.get("family", 0.1)

    for batch_idx, (images, species_ids, genus_ids, family_ids) in enumerate(loader):
        images = images.to(device)
        species_ids = species_ids.to(device)
        genus_ids = genus_ids.to(device)
        family_ids = family_ids.to(device)

        optimizer.zero_grad()

        # Forward pass returns a dictionary of logits
        outputs = model(images)

        # Calculate loss for each head
        loss_species = criterion(outputs["species"], species_ids)
        loss_genus = criterion(outputs["genus"], genus_ids)
        loss_family = criterion(outputs["family"], family_ids)

        # Weighted sum
        loss = (
            (loss_species * w_species)
            + (loss_genus * w_genus)
            + (loss_family * w_family)
        )

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def valid_one_epoch(model, loader, criterion, device, loss_weights):
    """
    Validates the model. Computes loss and Macro F1 score on the Species head.
    """
    model.eval()

    losses = AverageMeter()
    all_preds = []
    all_targets = []

    w_species = loss_weights.get("species", 1.0)
    w_genus = loss_weights.get("genus", 0.1)
    w_family = loss_weights.get("family", 0.1)

    with torch.no_grad():
        for batch_idx, (images, species_ids, genus_ids, family_ids) in enumerate(
            loader
        ):
            images = images.to(device)
            species_ids = species_ids.to(device)
            genus_ids = genus_ids.to(device)
            family_ids = family_ids.to(device)

            outputs = model(images)

            loss_species = criterion(outputs["species"], species_ids)
            loss_genus = criterion(outputs["genus"], genus_ids)
            loss_family = criterion(outputs["family"], family_ids)

            loss = (
                (loss_species * w_species)
                + (loss_genus * w_genus)
                + (loss_family * w_family)
            )
            losses.update(loss.item(), images.size(0))

            # For metrics, we only care about the primary task (Species)
            preds = torch.argmax(outputs["species"], dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(species_ids.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    score = get_score(all_targets, all_preds)

    return losses.avg, score


def fit(
    epochs=Config.EPOCHS,
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    label_smoothing=Config.LABEL_SMOOTHING,
    loss_weights=Config.LOSS_WEIGHTS,
    load_cached_data=True,
):
    """
    Main training loop.

    Args:
        epochs (int): Number of training epochs.
        lr (float): Max learning rate.
        weight_decay (float): Weight decay for optimizer.
        label_smoothing (float): Label smoothing factor for CrossEntropyLoss.
        loss_weights (dict): Weights for species, genus, and family losses.
        load_cached_data (bool): Whether to use cached hierarchy mappings.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    log_path = os.path.join(Config.WORKING_DIR, "train.log")
    logger = get_logger(log_path)

    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # 2. Data Loaders
    logger.info("Loading data...")
    train_loader, val_loader, _ = get_loaders(load_cached_data=load_cached_data)

    # 3. Model
    logger.info(f"Initializing model: {Config.BACKBONE}")
    model = get_model(pretrained=True, load_cached_hierarchy=load_cached_data)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=100.0,
    )

    # Loss Function (Label Smoothing applied to all heads)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 5. Training Loop
    best_score = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    logger.info("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, loss_weights
        )

        # Validate
        val_loss, val_score = valid_one_epoch(
            model, val_loader, criterion, device, loss_weights
        )

        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val F1: {val_score} - "
            f"Time: {elapsed:.0f}s"
        )

        # Save Best Model
        if val_score > best_score:
            logger.info(f"Score Improved: {best_score} -> {val_score}. Saving model...")
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    logger.info(f"Training complete. Best F1 Score: {best_score}")
    return best_score
