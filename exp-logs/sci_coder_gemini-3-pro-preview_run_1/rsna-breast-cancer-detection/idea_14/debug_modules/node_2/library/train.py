import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, Logger, probabilistic_f1
from library.data import get_dataloaders
from library.model import DeformableSiameseModel


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        target_img, contra_img, labels = batch

        target_img = target_img.to(device)
        contra_img = contra_img.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        logits = model(target_img, contra_img)
        loss = criterion(logits, labels)

        loss.backward()
        # Gradient clipping is explicitly disabled as per instructions
        optimizer.step()

        batch_size = target_img.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Performs validation and calculates pF1 score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            target_img, contra_img, labels = batch

            target_img = target_img.to(device)
            contra_img = contra_img.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(target_img, contra_img)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            batch_size = target_img.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_labels) > 0:
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        pf1 = probabilistic_f1(all_labels, all_probs)
    else:
        pf1 = 0.0

    return epoch_loss, pf1


def generate_submission(model_path, test_loader, device):
    """
    Generates predictions for the test set and creates the submission file.
    """
    print("Generating submission...")

    # Load Model
    model = DeformableSiameseModel(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_probs = []

    # Inference
    with torch.no_grad():
        for batch in test_loader:
            target_img, contra_img, _ = batch
            target_img = target_img.to(device)
            contra_img = contra_img.to(device)

            logits = model(target_img, contra_img)
            probs = torch.sigmoid(logits)

            # Flatten to 1D array
            all_probs.extend(probs.cpu().numpy().flatten())

    # Map predictions back to metadata
    # We access the dataframe directly from the dataset to ensure alignment
    # especially if subsampling (debug mode) was applied.
    df_test_aligned = test_loader.dataset.df.copy()

    if len(df_test_aligned) != len(all_probs):
        raise ValueError(
            f"Mismatch: Metadata rows {len(df_test_aligned)} != Predictions {len(all_probs)}"
        )

    df_test_aligned["cancer_prob"] = all_probs

    # Aggregate by prediction_id (take Max probability across views)
    submission_df = (
        df_test_aligned.groupby("prediction_id")["cancer_prob"].max().reset_index()
    )
    submission_df.rename(columns={"cancer_prob": "cancer"}, inplace=True)

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    """
    Main execution function.
    """
    # 1. Setup
    set_seed(Config.SEED)
    logger = Logger("train_log.txt")
    logger.log("Starting training run...")
    Config.print_config()

    device = torch.device(Config.DEVICE)

    # 2. Data
    # load_cached_data=True allows using pre-processed parquet files if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = DeformableSiameseModel(pretrained=True)
    model = model.to(device)

    # 4. Loss & Optimizer
    # Positive weight for class imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Training Loop
    best_pf1 = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        logger.log(f"Epoch {epoch+1}/{Config.EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Log metrics with full precision
        logger.log_metrics(epoch + 1, train_loss, val_loss, val_pf1)

        # Save Best Model (based on pF1)
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), best_model_path)
            logger.log(f"New best model saved! (pF1: {best_pf1})")

    logger.log(f"Training finished. Best Val pF1: {best_pf1}")

    # 6. Submission
    if os.path.exists(best_model_path):
        generate_submission(best_model_path, test_loader, device)
    else:
        logger.log("Error: Best model file not found. Skipping submission.")
