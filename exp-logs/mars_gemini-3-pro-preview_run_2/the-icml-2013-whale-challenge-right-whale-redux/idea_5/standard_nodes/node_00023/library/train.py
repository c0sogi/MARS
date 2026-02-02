import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint, calculate_roc_auc
from library.dataset import get_dataloaders
from library.model import WhaleEfficientNet


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
        targets = targets.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions and targets for AUC
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


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
            targets = targets.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(torch.sigmoid(logits).cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    val_auc = calculate_roc_auc(all_targets, all_preds)

    return val_loss, val_auc


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_clips = []
    all_probs = []

    with torch.no_grad():
        for inputs, clips in loader:
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_clips.extend(clips)
            all_probs.extend(probs)

    return all_clips, all_probs


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = WhaleEfficientNet(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # 4. Optimization Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 5. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.EPOCHS} - "
            f"Time: {elapsed:.2f}s - "
            f"LR: {current_lr:.6f} - "
            f"Train Loss: {train_loss:.10f} - "
            f"Train AUC: {train_auc:.10f} - "
            f"Val Loss: {val_loss:.10f} - "
            f"Val AUC: {val_auc:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_val_auc:
            print(
                f"Validation AUC improved from {best_val_auc:.10f} to {val_auc:.10f}. Saving model..."
            )
            best_val_auc = val_auc
            save_checkpoint(model, optimizer, epoch, val_auc, path=Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation AUC did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("\nRunning inference on test set with best model...")

    # Load best model
    checkpoint = load_checkpoint(model, path=Config.MODEL_PATH, device=Config.DEVICE)
    if checkpoint:
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} with Val AUC: {checkpoint['val_score']:.10f}"
        )
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    clips, probs = predict(model, test_loader, device)

    # 7. Submission
    submission_df = pd.DataFrame({"clip": clips, "probability": probs})

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(submission_df)}")
