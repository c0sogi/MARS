import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import (
    seed_everything,
    calculate_macro_f1,
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import get_dataloaders
from library.model import AnimalClassifier


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    num_samples = 0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

        preds = torch.argmax(outputs, dim=1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / num_samples if num_samples > 0 else 0.0
    epoch_f1 = calculate_macro_f1(np.array(all_targets), np.array(all_preds))

    return epoch_loss, epoch_f1


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    num_samples = 0

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_loss = running_loss / num_samples if num_samples > 0 else 0.0
    val_f1 = calculate_macro_f1(np.array(all_targets), np.array(all_preds))

    return val_loss, val_f1


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    TTA involves averaging predictions from the original image and a horizontally flipped version.
    """
    model.eval()
    ids = []
    predictions = []

    print("Generating submission with Test-Time Augmentation (TTA)...")

    with torch.no_grad():
        for images, _, image_ids in loader:
            images = images.to(device)

            # 1. Forward pass with original images
            outputs_orig = model(images)

            # 2. Forward pass with horizontally flipped images (TTA)
            images_flipped = torch.flip(images, dims=[3])  # N, C, H, W -> flip W
            outputs_flip = model(images_flipped)

            # 3. Average logits
            outputs_avg = (outputs_orig + outputs_flip) / 2.0

            # 4. Get predictions
            preds = torch.argmax(outputs_avg, dim=1).cpu().numpy()

            ids.extend(image_ids)
            predictions.extend(preds)

    # Create submission DataFrame
    df = pd.DataFrame({"Id": ids, "Predicted": predictions})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(num_epochs=Config.NUM_EPOCHS, patience=5):
    """
    Main function to orchestrate training, validation, and submission generation.
    """
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Initialize Model
    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = AnimalClassifier(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    # OneCycleLR requires total steps
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Training Loop Variables
    best_val_f1 = -1.0
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validate
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics (Full precision as requested)
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}, Train F1: {train_f1}")
        print(f"Val Loss: {val_loss}, Val F1: {val_f1}")

        # Checkpoint & Early Stopping
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            patience_counter = 0
            print("New best model found. Saving checkpoint.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_f1": best_val_f1,
            },
            is_best=is_best,
        )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation F1: {best_val_f1}")

    # Generate Submission using the best model
    print("Loading best model for submission...")
    checkpoint = load_checkpoint(Config.MODEL_CHECKPOINT_PATH, model)
    if checkpoint is not None:
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} with Val F1: {checkpoint['best_val_f1']}"
        )
    else:
        print("Warning: No checkpoint found. Using current model state.")

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
