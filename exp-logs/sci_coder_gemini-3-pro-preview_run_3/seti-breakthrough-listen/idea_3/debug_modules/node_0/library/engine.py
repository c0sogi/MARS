import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion, get_score
from library.model import Shallow3DCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        batch_size = images.size(0)
        dataset_size += batch_size

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    return running_loss / dataset_size


def valid_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds = []
    valid_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            batch_size = images.size(0)
            dataset_size += batch_size

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size

            preds.append(torch.sigmoid(outputs).cpu().numpy())
            valid_targets.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    valid_targets = np.concatenate(valid_targets)
    auc = get_score(valid_targets, preds)

    return running_loss / dataset_size, auc


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (Horizontal Flip).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # Forward pass 1: Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # Forward pass 2: Horizontal Flip (Frequency dimension is last)
            # Input shape: (B, 1, Depth, Height, Width)
            images_flip = torch.flip(images, dims=[-1])
            out_flip = model(images_flip)
            prob_flip = torch.sigmoid(out_flip)

            # Average probabilities
            avg_prob = (prob_orig + prob_flip) / 2.0

            preds.append(avg_prob.cpu().numpy())

    return np.concatenate(preds)


def fit(
    train_loader, val_loader, test_loader, epochs=Config.EPOCHS, device=Config.DEVICE
):
    """
    Orchestrates the training, validation, and inference pipeline with Early Stopping.
    """
    print(f"Using device: {device}")

    # Initialize Model, Optimizer, Scheduler, Loss
    model = Shallow3DCNN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop variables
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
            # print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")

    # --- Inference ---
    print("Generating predictions for test set...")

    # Load best model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: No model file found. Using current model state.")

    # Predict with TTA
    test_preds = predict_with_tta(model, test_loader, device)

    # Save Submission
    df_sub = pd.read_csv(Config.TEST_METADATA)
    df_sub["target"] = test_preds

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_sub[["id", "target"]].to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
