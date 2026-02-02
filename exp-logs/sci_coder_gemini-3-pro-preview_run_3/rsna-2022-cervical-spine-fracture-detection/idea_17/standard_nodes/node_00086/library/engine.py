import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import seed_everything, RSNALoss
from library.dataset import RSNADataset
from library.model import FractureModel


def train_one_epoch(model, optimizer, scheduler, scaler, dataloader, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = RSNALoss()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass (returns logits)
        with autocast():
            logits = model(images)
            loss = criterion(logits, targets)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient Clipping
        if Config.MAX_GRAD_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    # Step the scheduler (Epoch-based)
    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    """
    Validates the model for one epoch.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    criterion = RSNALoss()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            batch_size = images.size(0)

            # Forward pass
            logits = model(images)

            # Compute loss
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def fit_model():
    """
    Main training loop. Initializes model, data, optimizer, and runs epochs.
    Saves the best model based on validation loss.
    """
    seed_everything(Config.SEED)

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    # --- Data Loading ---
    train_dataset = RSNADataset(subset="train")
    val_dataset = RSNADataset(subset="val")

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

    # --- Model Setup ---
    device = Config.DEVICE
    model = FractureModel(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    )
    model = model.to(device)

    # --- Optimization ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scaler = GradScaler()

    # Scheduler: Cosine Annealing
    t_max = int(Config.T_MAX_MULT * Config.EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=Config.MIN_LR
    )

    best_loss = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, optimizer, scheduler, scaler, train_loader, device, epoch
        )
        val_loss = valid_one_epoch(model, val_loader, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpoint (Early Stopping Logic)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with loss: {val_loss}")

        # Memory Cleanup
        gc.collect()
        torch.cuda.empty_cache()


def inference_and_submit():
    """
    Loads the best model, runs inference on the test set, and generates the submission file.
    """
    seed_everything(Config.SEED)

    # --- Data Loading ---
    test_dataset = RSNADataset(subset="test")

    # Load metadata to ensure alignment of StudyInstanceUIDs
    # Use the dataset's internal dataframe to ensure consistency (especially in debug mode)
    study_ids = test_dataset.df["StudyInstanceUID"].values
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model Loading ---
    device = Config.DEVICE
    model = FractureModel(
        backbone_name=Config.BACKBONE, pretrained=False, num_classes=Config.NUM_CLASSES
    )

    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_SAVE_PATH}")
    else:
        print("Warning: No trained model found. Using random initialization.")

    model = model.to(device)
    model.eval()

    all_preds = []

    # --- Inference Loop ---
    print("Running inference on test set...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)

            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        all_preds = np.zeros((0, 8))

    # --- Submission Formatting ---
    # Map model outputs (0-7) to submission row suffixes
    column_map = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    submission_rows = []

    # Ensure we have predictions for every study in the metadata
    if len(study_ids) != len(all_preds):
        print(
            f"Error: Mismatch between studies ({len(study_ids)}) and predictions ({len(all_preds)})"
        )

    for idx, study_id in enumerate(study_ids):
        preds = all_preds[idx]  # Shape (8,)

        for class_idx, class_name in enumerate(column_map):
            row_id = f"{study_id}_{class_name}"
            prob = preds[class_idx]
            submission_rows.append({"row_id": row_id, "fractured": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Save to disk
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
