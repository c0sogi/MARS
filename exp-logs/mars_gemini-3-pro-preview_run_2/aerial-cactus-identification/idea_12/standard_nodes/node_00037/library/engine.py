import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    SEEDS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    T_MAX,
    ETA_MIN,
)
from library.utils import AverageMeter, compute_auc, seed_everything
from library.model import NarrowSEResNet
from library.dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

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
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(logits)

            losses.update(loss.item(), images.size(0))
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    auc = compute_auc(np.array(all_targets), np.array(all_preds))
    return losses.avg, auc


def train_model(seed, train_loader, val_loader, device):
    """
    Instantiates and trains a model for a specific seed.
    Saves the best model based on Validation AUC.
    """
    seed_everything(seed)

    print(f"\n--- Training Seed {seed} ---")

    # Initialize Model
    model = NarrowSEResNet()
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=ETA_MIN
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved for seed {seed} with AUC: {best_auc}")

    print(f"Finished Seed {seed}. Best Val AUC: {best_auc}")
    return best_model_path


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (Original, H-Flip, V-Flip).
    Returns averaged probabilities and image IDs.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, _ in loader:
            # Loader returns (image, label), but we also need IDs which are stored in dataset
            # However, the loader iteration gives tensors. We'll collect IDs separately or
            # rely on the fact that the loader is not shuffled for test.
            # The standard approach with the provided Dataset class:
            # The dataset has .ids attribute, and loader preserves order if shuffle=False.
            pass

        # To ensure alignment, we iterate and rely on the loader order (shuffle=False)
        # and fetch IDs from the dataset directly after the loop, or assume index alignment.
        # Let's just iterate images.

        for i, (images, _) in enumerate(loader):
            images = images.to(device)

            # 1. Original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip
            images_h = torch.flip(images, [3])
            logits_h = model(images_h)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip
            images_v = torch.flip(images, [2])
            logits_v = model(images_v)
            probs_v = torch.sigmoid(logits_v)

            # Average probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            all_preds.extend(avg_probs.cpu().numpy().flatten())

    return np.array(all_preds)


def generate_submission():
    """
    Main execution function.
    1. Trains models for all seeds.
    2. Generates predictions using TTA and Ensemble Averaging.
    3. Saves submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Train Ensemble
    model_paths = []
    for seed in SEEDS:
        path = train_model(seed, train_loader, val_loader, device)
        model_paths.append(path)

    # Inference & Ensemble Aggregation
    print("\n--- Starting Inference ---")

    # Initialize array to store sum of predictions
    num_test_samples = len(test_loader.dataset)
    ensemble_preds = np.zeros(num_test_samples)

    for path in model_paths:
        print(f"Predicting with model: {path}")
        model = NarrowSEResNet()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)

        preds = predict_with_tta(model, test_loader, device)
        ensemble_preds += preds

    # Average over seeds
    final_preds = ensemble_preds / len(SEEDS)

    # Create Submission DataFrame
    # Retrieve IDs from the test dataset
    test_ids = test_loader.dataset.ids

    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())
