import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, get_score, set_seed
from library.dataset import get_dataloaders
from library.model import LateFusionModel


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Runs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        probs = torch.sigmoid(logits)
        losses.update(loss.item(), inputs.size(0))

        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(probs.detach().cpu().numpy())

    epoch_score = get_score(all_targets, all_preds)
    return losses.avg, epoch_score


def validate(model, dataloader, criterion, device):
    """
    Runs validation on the provided dataloader.
    """
    model.eval()
    losses = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            losses.update(loss.item(), inputs.size(0))
            all_targets.extend(targets.detach().cpu().numpy())
            all_preds.extend(probs.detach().cpu().numpy())

    epoch_score = get_score(all_targets, all_preds)
    return losses.avg, epoch_score


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
    debug=Config.DEBUG,
    num_workers=Config.NUM_WORKERS,
):
    """
    Main function to train the model, perform validation with early stopping,
    and generate the submission file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # --- Data Loading ---
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers, debug=debug
    )

    # --- Model Initialization ---
    model = LateFusionModel(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # --- Optimization ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # --- Training Loop ---
    best_score = 0.0
    patience_counter = 0
    best_model_path = Config.MODEL_PATH

    # Ensure working directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_score = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Printing full precision as requested
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Train AUC: {train_score} | Val Loss: {val_loss} | Val AUC: {val_score}"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best Model Saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # --- Inference ---
    print("Starting inference on test set...")

    # Load best model
    # Use pretrained=False for inference to avoid reloading ImageNet weights
    model = LateFusionModel(pretrained=False)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found, using current model weights.")

    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            predictions.extend(probs)

    # --- Submission ---
    # Retrieve IDs from the test dataset metadata
    test_ids = test_loader.dataset.metadata["id"].values

    # Ensure lengths match (robustness check)
    if len(predictions) != len(test_ids):
        print(
            f"Warning: Prediction count ({len(predictions)}) differs from ID count ({len(test_ids)}). Truncating to minimum."
        )
        min_len = min(len(predictions), len(test_ids))
        predictions = predictions[:min_len]
        test_ids = test_ids[:min_len]

    submission = pd.DataFrame({"id": test_ids, "target": predictions})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
