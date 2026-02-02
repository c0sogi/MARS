import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import timm
import numpy as np

from library.config import Config
from library.utils import probabilistic_f1
from library.dataset import get_dataloaders


class MetadataEfficientNet(nn.Module):
    """
    EfficientNet-B0 based model that accepts fused Image + Metadata inputs.
    """

    def __init__(self, backbone_name=Config.BACKBONE, pretrained=True):
        super(MetadataEfficientNet, self).__init__()

        # Initialize EfficientNet with modified input channels
        # in_chans = Config.TOTAL_INPUT_CHANNELS (1 image + 2 metadata = 3)
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            in_chans=Config.TOTAL_INPUT_CHANNELS,
        )

    def forward(self, x):
        # x shape: (Batch_Size, 3, 512, 512)
        return self.model(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        # Gradient clipping to handle instability from high pos_weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Collect predictions for pF1 calculation
        probs = torch.sigmoid(logits).detach().cpu()
        all_targets.append(targets.cpu())
        all_preds.append(probs)

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)

    # Calculate pF1
    epoch_pf1 = probabilistic_f1(all_targets, all_preds)

    return epoch_loss, epoch_pf1


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).cpu()
            all_targets.append(targets.cpu())
            all_preds.append(probs)

    val_loss = running_loss / len(loader.dataset)

    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)

    val_pf1 = probabilistic_f1(all_targets, all_preds)

    return val_loss, val_pf1


def run_training(sample_size=Config.SAMPLE_SIZE, epochs=Config.EPOCHS):
    """
    Main training loop with Early Stopping.
    """
    # Ensure reproducibility
    Config.set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Load Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=True, sample_size=sample_size
    )

    # Initialize Model
    model = MetadataEfficientNet().to(device)

    # Loss Function (Weighted BCE)
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # Early Stopping Configuration
    best_pf1 = -1.0
    patience = 3  # Stop if no improvement for 3 epochs
    counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_pf1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.1f}s")
        print(f"  Train Loss: {train_loss:.6f} | Train pF1: {train_pf1:.6f}")
        # Print Val pF1 with full precision as requested
        print(f"  Val Loss:   {val_loss:.6f} | Val pF1:   {val_pf1}")

        # Save Best Model
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  [New Best Model Saved] pF1: {val_pf1}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val pF1: {best_pf1}")
    return model


def generate_submission(model=None, sample_size=None):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    device = torch.device(Config.DEVICE)

    # Load Test Data
    _, _, test_loader = get_dataloaders(load_cached_data=True, sample_size=sample_size)

    # Load Model if not provided
    if model is None:
        model = MetadataEfficientNet().to(device)
        if os.path.exists(Config.MODEL_SAVE_PATH):
            model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=device)
            )
            print(f"Loaded model from {Config.MODEL_SAVE_PATH}")
        else:
            print(
                "Warning: No trained model found at default path. Using random weights."
            )

    model.eval()

    predictions = []
    ids = []

    print("Generating predictions on test set...")

    with torch.no_grad():
        for inputs, pred_ids in test_loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            ids.extend(pred_ids)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"prediction_id": ids, "cancer": predictions})

    # Aggregate predictions by prediction_id (taking the max probability)
    df_sub = df_sub.groupby("prediction_id", as_index=False)["cancer"].max()

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
