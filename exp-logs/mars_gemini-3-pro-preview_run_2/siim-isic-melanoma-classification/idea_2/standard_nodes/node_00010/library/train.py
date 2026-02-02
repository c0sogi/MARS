import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, AverageMeter, get_roc_auc
from library.loss import FocalLoss
from library.data import get_dataloaders
from library.model import DeepHybridEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    all_targets = []
    all_preds = []

    for images, meta, targets in loader:
        images = images.to(device)
        meta = meta.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, meta)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Update metrics
        loss_meter.update(loss.item(), images.size(0))

        # Collect predictions for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    epoch_auc = get_roc_auc(all_targets, all_preds)
    return loss_meter.avg, epoch_auc


def valid_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, meta, targets in loader:
            images = images.to(device)
            meta = meta.to(device)
            targets = targets.to(device)

            logits = model(images, meta)
            loss = criterion(logits, targets)

            loss_meter.update(loss.item(), images.size(0))

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    epoch_auc = get_roc_auc(all_targets, all_preds)
    return loss_meter.avg, epoch_auc


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []
    image_names = []

    # Retrieve image names from the dataset
    # The loader wraps the dataset, which has the dataframe
    dataset_df = loader.dataset.df
    image_names = dataset_df["image_name"].values

    with torch.no_grad():
        for images, meta, _ in loader:
            images = images.to(device)
            meta = meta.to(device)

            logits = model(images, meta)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs.flatten())

    return image_names, np.array(all_preds)


def run_training():
    """
    Main orchestration function.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # Determine metadata dimension dynamically
    dummy_img, dummy_meta, _ = next(iter(train_loader))
    meta_dim = dummy_meta.shape[1]
    print(f"Detected Metadata Dimension: {meta_dim}")

    # 3. Model Initialization
    print(f"Initializing {Config.MODEL_NAME}...")
    model = DeepHybridEfficientNet(meta_dim=meta_dim).to(device)

    # 4. Optimizer, Loss, Scheduler
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler setup
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(len(train_loader) * Config.WARMUP_EPOCHS)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"  Train Loss: {train_loss:.10f} | Train AUC: {train_auc:.10f}")
        print(f"  Val Loss:   {val_loss:.10f} | Val AUC:   {val_auc:.10f}")

        # Early Stopping & Checkpointing
        if val_auc > best_auc + Config.EARLY_STOPPING_MIN_DELTA:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"  New best model saved! (AUC: {best_auc:.10f})")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        )
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    print("Generating predictions on test set...")
    image_names, predictions = predict_test(model, test_loader, device)

    # 7. Submission
    submission_df = pd.DataFrame({"image_name": image_names, "target": predictions})

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())


if __name__ == "__main__":
    run_training()
