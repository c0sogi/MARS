import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.model import SteerableCactusNet
from library.data import get_dataloaders, mixup_data, mixup_criterion


def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    dataset_size = len(loader.dataset)

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, Config.MIXUP_ALPHA, device
            )
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / dataset_size


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / dataset_size

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        auc = calculate_roc_auc(all_targets, all_preds)
    else:
        auc = 0.0

    return avg_loss, auc


def predict_tta(model, loader, device):
    """
    Performs inference using 4-view Test Time Augmentation.
    Views: Original, Horizontal Flip, Vertical Flip, Rotate 180.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for inputs, ids in loader:
            inputs = inputs.to(device)

            # 1. Original
            out1 = torch.sigmoid(model(inputs))

            # 2. Horizontal Flip
            out2 = torch.sigmoid(model(torch.flip(inputs, [3])))

            # 3. Vertical Flip
            out3 = torch.sigmoid(model(torch.flip(inputs, [2])))

            # 4. Rotate 180 (equivalent to flip dim 2 then dim 3)
            out4 = torch.sigmoid(model(torch.rot90(inputs, k=2, dims=[2, 3])))

            # Average predictions
            avg_preds = (out1 + out2 + out3 + out4) / 4.0

            all_preds.append(avg_preds.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds).flatten(), all_ids


def train_engine():
    """
    Main training engine.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders()

    # Initialize Model
    model = SteerableCactusNet().to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = 0.0
    patience = 10
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs.")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_mixup=Config.USE_MIXUP,
        )

        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Checkpointing and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print(f"  New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")


def generate_submission_csv():
    """
    Generates the submission CSV using the best trained model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders()

    # Load Model
    model = SteerableCactusNet().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print(
            f"Warning: Model file not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    # Predict
    probs, ids = predict_tta(model, test_loader, device)

    # Save
    df = pd.DataFrame({"id": ids, "has_cactus": probs})

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
