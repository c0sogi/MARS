import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEEDS,
    CHANNELS,
    NUM_WORKERS,
    DEVICE,
    WORKING_DIR,
    SUBMISSION_DIR,
)
from library.utils import set_seed, calculate_roc_auc
from library.dataset import CactusDataset, get_transforms, get_data_arrays
from library.model import MultiScaleResNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # Apply sigmoid for AUC calculation
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = calculate_roc_auc(all_targets, all_preds)
    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Performs validation on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

    val_loss = running_loss / len(loader.dataset)
    val_auc = calculate_roc_auc(all_targets, all_preds)
    return val_loss, val_auc


def predict_tta(model, images_np, device, batch_size=BATCH_SIZE):
    """
    Performs Test Time Augmentation (TTA) inference.
    Averages predictions from: Original, Horizontal Flip, Vertical Flip.
    """
    model.eval()
    transform = get_transforms(mode="test")
    num_images = len(images_np)
    all_preds = []

    # Process in batches to manage memory
    for i in range(0, num_images, batch_size):
        batch_imgs = images_np[i : i + batch_size]

        # Prepare TTA batches
        # 1. Original
        t1 = torch.stack([transform(img) for img in batch_imgs]).to(device)

        # 2. Horizontal Flip
        t2 = torch.stack([transform(cv2.flip(img, 1)) for img in batch_imgs]).to(device)

        # 3. Vertical Flip
        t3 = torch.stack([transform(cv2.flip(img, 0)) for img in batch_imgs]).to(device)

        with torch.no_grad():
            p1 = torch.sigmoid(model(t1))
            p2 = torch.sigmoid(model(t2))
            p3 = torch.sigmoid(model(t3))

        # Average predictions
        avg_p = (p1 + p2 + p3) / 3.0
        all_preds.extend(avg_p.cpu().numpy().flatten())

    return np.array(all_preds)


def run_experiment(load_cached_data=True):
    """
    Main driver function for the experiment.
    Trains models across multiple seeds and generates an averaged submission.
    """
    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 1. Load Data
    print("Loading data...")
    data = get_data_arrays(load_cached_data=load_cached_data)

    train_images_all = data["train_images"]
    train_labels_all = data["train_labels"]
    val_images_all = data["val_images"]
    val_labels_all = data["val_labels"]
    test_images = data["test_images"]
    test_ids = data["test_ids"]

    # Array to store accumulated predictions from all seeds
    final_test_preds = np.zeros(len(test_ids))

    # 2. Training Loop per Seed
    for seed in SEEDS:
        print(f"\n================ Training Seed {seed} ================")
        set_seed(seed)

        # Prepare Datasets and Loaders
        train_ds = CactusDataset(
            train_images_all, train_labels_all, transform=get_transforms(mode="train")
        )
        val_ds = CactusDataset(
            val_images_all, val_labels_all, transform=get_transforms(mode="val")
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model, Optimizer, Scheduler
        model = MultiScaleResNet(channels=CHANNELS).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.BCEWithLogitsLoss()

        # Training Variables
        best_val_auc = 0.0
        patience = 5
        no_improve = 0
        best_model_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")

        # Epoch Loop
        for epoch in range(EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE
            )
            val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

            scheduler.step()

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f}"
            )

            # Checkpointing & Early Stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # 3. Inference for current seed
        print(f"Loading best model for Seed {seed} and generating TTA predictions...")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        seed_preds = predict_tta(model, test_images, DEVICE)
        final_test_preds += seed_preds

    # 4. Aggregate and Save Submission
    final_test_preds /= len(SEEDS)

    sub_df = pd.DataFrame({"id": test_ids, "has_cactus": final_test_preds})

    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(save_path, index=False)
    print(f"\nSubmission saved successfully to {save_path}")
