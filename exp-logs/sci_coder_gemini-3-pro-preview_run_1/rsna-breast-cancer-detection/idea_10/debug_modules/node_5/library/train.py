import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import PyramidSymmetryDifferenceModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Move data to device
        img_target = batch["image"].to(device)
        img_contra = batch["contra_image"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)  # (B, 1)

        batch_size = img_target.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(img_target, img_contra)

        # Loss calculation
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Optimizer step (No gradient clipping as per requirements)
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Probabilistic F1 score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            img_target = batch["image"].to(device)
            img_contra = batch["contra_image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            batch_size = img_target.size(0)

            logits = model(img_target, img_contra)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = labels.cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets)

    avg_loss = running_loss / dataset_size

    # Concatenate all batches
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Calculate pF1
    pf1 = probabilistic_f1(all_targets, all_probs)

    return avg_loss, pf1


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            img_target = batch["image"].to(device)
            img_contra = batch["contra_image"].to(device)
            prediction_ids = batch[Config.ID_COL]

            logits = model(img_target, img_contra)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for pid, prob in zip(prediction_ids, probs):
                results.append({Config.ID_COL: pid, Config.TARGET_COL: prob})

    return pd.DataFrame(results)


def run_training(debug=Config.DEBUG, epochs=Config.NUM_EPOCHS):
    """
    Main execution function for training and submission generation.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading Datasets...")
    # get_dataloaders handles caching internally
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = PyramidSymmetryDifferenceModel().to(device)

    # 4. Optimization Setup
    # Use weighted BCE for class imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training Loop
    best_pf1 = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val pF1: {val_pf1}"
        )

        # Save best model
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with pF1: {best_pf1}")

    # 6. Inference
    print("Starting Inference on Test Set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model weights.")
    else:
        print("Warning: Best model not found. Using current weights.")

    df_preds = predict_test(model, test_loader, device)

    # 7. Submission Aggregation
    # Group by prediction_id (breast level) and take the Max probability across views
    submission = df_preds.groupby(Config.ID_COL)[Config.TARGET_COL].max().reset_index()

    # Save submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
