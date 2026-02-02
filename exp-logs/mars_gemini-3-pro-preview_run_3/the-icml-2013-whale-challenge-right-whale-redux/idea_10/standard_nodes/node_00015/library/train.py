import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from library.config import Config
from library.utils import set_seed, mixup_criterion, calculate_auc
from library.dataset import get_dataloaders, get_test_loader
from library.models import WhaleClassifier


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup augmentation and weighted BCE loss.
    """
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Mixup Logic
        if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
            # Generate mixup lambda
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)

            # Shuffle indices for mixing
            index = torch.randperm(images.size(0)).to(device)

            # Mix inputs
            mixed_images = lam * images + (1 - lam) * images[index]

            # Forward pass
            outputs = model(mixed_images)

            # Calculate mixed loss
            # labels are (B,), outputs are (B, 1) usually, need to ensure shapes match for BCE
            # BCEWithLogitsLoss expects target to have same shape as input
            labels_expanded = labels.view(-1, 1)
            loss = mixup_criterion(
                criterion, outputs, labels_expanded, labels_expanded[index], lam
            )
        else:
            # Standard training
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1))

        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set and computes AUC.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1))

            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

    avg_loss = running_loss / len(loader)
    auc = calculate_auc(all_targets, all_preds)

    return avg_loss, auc


def run_training(load_cached_data=True):
    """
    Main training pipeline: Setup, Training Loop, Validation, and Checkpointing.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader = get_dataloaders(load_cached_data=load_cached_data)

    # Calculate Class Weights for Loss
    pos_weight = None
    if Config.USE_CLASS_WEIGHTS:
        # Access labels from the dataset
        train_labels = train_loader.dataset.labels.numpy()
        num_pos = np.sum(train_labels == 1)
        num_neg = np.sum(train_labels == 0)

        # Avoid division by zero
        if num_pos > 0:
            weight_val = num_neg / num_pos
            pos_weight = torch.tensor([weight_val]).to(device)
            print(f"Using positive class weight: {weight_val}")

    # 3. Model & Loss
    model = WhaleClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Linear Warmup + Cosine Annealing
    # Note: T_max for Cosine is epochs - warmup
    scheduler_warmup = LinearLR(
        optimizer, start_factor=0.01, total_iters=Config.WARMUP_EPOCHS
    )
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS - Config.WARMUP_EPOCHS, eta_min=Config.MIN_LR
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[Config.WARMUP_EPOCHS],
    )

    # 5. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging (Full precision as requested)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")


def generate_submission(load_cached_data=True):
    """
    Loads the best model and generates predictions for the test set.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    test_loader, clip_names = get_test_loader(load_cached_data=load_cached_data)

    # Load Model
    model = WhaleClassifier().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best model not found at {Config.BEST_MODEL_PATH}. Run training first."
        )

    print(f"Loading model from {Config.BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    all_probs = []

    print("Generating predictions...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy().flatten())

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"clip": clip_names, "probability": all_probs})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_sub.head())
