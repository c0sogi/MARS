import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import (
    SEEDS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    WORKING_DIR,
    SUBMISSION_PATH,
    NUM_CLASSES,
    TTA_ENABLED,
)
from library.utils import set_seed, EarlyStopping
from library.dataset import get_dataloaders
from library.model import CactusNet, train_one_epoch, validate


def predict_tta(models, loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA) and Model Ensembling.

    Args:
        models (list): List of loaded PyTorch models (one for each seed).
        loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        dict: Dictionary mapping image IDs to average predicted probabilities.
    """
    # Ensure all models are in evaluation mode
    for m in models:
        m.eval()

    predictions = {}

    with torch.no_grad():
        for images, _, ids in loader:
            images = images.to(device)
            batch_preds = []

            # TTA Views: Original, Horizontal Flip, Vertical Flip
            views = [images]
            if TTA_ENABLED:
                views.append(torch.flip(images, [3]))  # Horizontal
                views.append(torch.flip(images, [2]))  # Vertical

            # Aggregate predictions across all views and all models
            for view in views:
                for model in models:
                    logits = model(view)
                    probs = torch.sigmoid(logits)
                    batch_preds.append(probs.cpu().numpy())

            # Average predictions for this batch
            # Shape: (num_views * num_models, batch_size, 1)
            batch_preds = np.array(batch_preds)
            # Mean over the first dimension (ensembling + TTA)
            avg_preds = np.mean(batch_preds, axis=0).flatten()

            for img_id, pred in zip(ids, avg_preds):
                predictions[img_id] = pred

    return predictions


def run_training():
    """
    Executes the full training pipeline:
    1. Loops through defined SEEDS.
    2. Trains a model for each seed with Early Stopping.
    3. Saves the best model for each seed.
    4. Generates predictions using TTA and Ensembling.
    5. Saves the submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Get DataLoaders (cached)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    model_paths = []

    # --- Training Phase ---
    for seed in SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        set_seed(seed)

        model = CactusNet(num_classes=NUM_CLASSES).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

        model_save_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")
        early_stopping = EarlyStopping(
            patience=PATIENCE, verbose=True, path=model_save_path
        )

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc}"
            )

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        model_paths.append(model_save_path)

    # --- Inference Phase ---
    print("\n--- Generating Submission with TTA and Ensembling ---")

    # Load all best models
    models = []
    for path in model_paths:
        m = CactusNet(num_classes=NUM_CLASSES).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        models.append(m)

    # Generate predictions
    preds_map = predict_tta(models, test_loader, device)

    # Create Submission DataFrame
    submission_data = [{"id": k, "has_cactus": v} for k, v in preds_map.items()]
    df_sub = pd.DataFrame(submission_data)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save Submission
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
