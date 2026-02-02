import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, get_logger, compute_auc
from library.dataset import TumorDataset, get_transforms
from library.model import ConvNeXtTinyCustom

# Initialize Logger
logger = get_logger("engine")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape (B, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    auc_score = compute_auc(all_labels, all_preds)

    return epoch_loss, auc_score


def predict_tta(model, loader, device):
    """
    Performs inference with Test Time Augmentation (TTA).
    Augmentations: Original, HFlip, VFlip, Rotate90.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)

            # 1. Original
            logits_1 = model(images)
            probs_1 = torch.sigmoid(logits_1)

            # 2. Horizontal Flip
            images_hflip = torch.flip(images, [3])
            logits_2 = model(images_hflip)
            probs_2 = torch.sigmoid(logits_2)

            # 3. Vertical Flip
            images_vflip = torch.flip(images, [2])
            logits_3 = model(images_vflip)
            probs_3 = torch.sigmoid(logits_3)

            # 4. Rotate 90 degrees
            images_rot = torch.rot90(images, 1, [2, 3])
            logits_4 = model(images_rot)
            probs_4 = torch.sigmoid(logits_4)

            # Average probabilities
            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            # Store results
            # We need to map these back to IDs.
            # The loader iterates sequentially, so we can index into the dataframe later
            # or assume the loader returns items in order.
            results.append(avg_probs.cpu().numpy())

    return np.concatenate(results)


def run():
    """
    Main execution function for training and inference.
    """
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    logger.info("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Val size: {len(val_df)}")
    logger.info(f"Test size: {len(test_df)}")

    # Create Datasets
    train_dataset = TumorDataset(train_df, transforms=get_transforms("train"))
    val_dataset = TumorDataset(val_df, transforms=get_transforms("val"))
    test_dataset = TumorDataset(test_df, transforms=get_transforms("test"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 2. Model Setup
    # -------------------------------------------------------------------------
    logger.info(f"Initializing model: {Config.MODEL_NAME}")
    model = ConvNeXtTinyCustom(pretrained=Config.PRETRAINED)
    model.to(Config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # -------------------------------------------------------------------------
    # 3. Training Loop
    # -------------------------------------------------------------------------
    logger.info("Starting training...")
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Early Stopping & Model Saving
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved with AUC: {best_auc:.10f}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    # -------------------------------------------------------------------------
    # 4. Inference
    # -------------------------------------------------------------------------
    logger.info("Loading best model for inference...")
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )

    logger.info("Generating predictions on test set (TTA Enabled)...")
    predictions = predict_tta(model, test_loader, Config.DEVICE)

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    submission_df = pd.DataFrame({"id": test_df["id"], "label": predictions.flatten()})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info("Done.")


if __name__ == "__main__":
    run()
