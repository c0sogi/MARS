import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import timm
from library.config import Config
from library.utils import mixup_data, mixup_criterion, get_score


class TechnoModel(nn.Module):
    """
    2D CNN (ResNet18d) that treats the 6 cadence observations as input channels.
    Input shape: (Batch, 6, Height=273, Width=256)
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=True):
        super(TechnoModel, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=1,
        )

    def forward(self, x):
        return self.model(x)


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
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

    # Step scheduler if provided (CosineAnnealingLR usually per epoch)
    if scheduler:
        scheduler.step()

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
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


def predict(model, loader, device):
    """
    Generates predictions using TTA (Test Time Augmentation).
    TTA Strategy: Average prediction of original and horizontally flipped input.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # Forward pass 1: Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # Forward pass 2: Horizontal Flip (Time dimension is last: W)
            # Input shape is (B, 6, H, W)
            images_flip = torch.flip(images, dims=[-1])
            out_flip = model(images_flip)
            prob_flip = torch.sigmoid(out_flip)

            # Average probabilities
            avg_prob = (prob_orig + prob_flip) / 2.0

            preds.append(avg_prob.cpu().numpy())

    return np.concatenate(preds)


def run(train_loader, val_loader, test_loader):
    """
    Orchestrates the training and inference pipeline.
    """
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Initialize Model
    model = TechnoModel().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Early Stopping & Model Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc:.6f}")

    # Inference on Test Set
    print("Generating predictions for test set...")
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH))
    else:
        print("Warning: No model file found. Using current model state.")

    test_preds = predict(model, test_loader, device)

    # Save Submission
    # Load test metadata to get IDs and preserve order
    df_sub = test_loader.dataset.df.copy()
    df_sub["target"] = test_preds

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save only required columns
    df_sub[["id", "target"]].to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
