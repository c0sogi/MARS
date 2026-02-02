import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import seed_everything, probabilistic_f1, apply_analytical_correction
from library.dataset import get_dataloaders
from library.model import BreastMILModel


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using AMP and Gradient Accumulation.
    """
    model.train()
    running_loss = 0.0

    # Initialize Scaler for AMP (Cite debug_lesson_2)
    scaler = GradScaler()

    optimizer.zero_grad()

    for batch_idx, (images, labels, _) in enumerate(loader):
        # Note: 'images' is a list of tensors. We pass them to the model directly.
        # The model handles moving them to GPU efficiently to save memory.

        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        # Forward pass with AMP
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
            # Normalize loss for gradient accumulation
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass with scaler
        scaler.scale(loss).backward()

        # Step optimizer every N steps
        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Track full loss for reporting (undo division)
        running_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS

    # Handle remaining gradients if last batch didn't trigger step
    if len(loader) % Config.GRADIENT_ACCUMULATION_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using Analytical Correction.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            # Pass list of images (CPU or GPU) to model
            labels = labels.to(device).unsqueeze(1)

            # Use autocast for evaluation to save memory
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item()

            # Apply Analytical Correction for evaluation metrics
            # This shifts the logits based on the difference between train prevalence (0.5)
            # and test prevalence (~0.02)
            probs = apply_analytical_correction(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Calculate Probabilistic F1
    pf1 = probabilistic_f1(all_labels, all_preds)
    avg_loss = running_loss / len(loader)

    return avg_loss, pf1


def predict_and_submit(model, loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating predictions for submission...")
    model.eval()

    prediction_ids = []
    probabilities = []

    with torch.no_grad():
        for images, _, ids in loader:
            # Use autocast for inference
            with autocast():
                logits = model(images)

            # Apply Analytical Correction for final predictions
            probs = apply_analytical_correction(logits)

            prediction_ids.extend(ids)
            probabilities.extend(probs.cpu().numpy().flatten())

    # Create DataFrame
    df_sub = pd.DataFrame({"prediction_id": prediction_ids, "cancer": probabilities})

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save submission
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_sub.head())


def run_training():
    """
    Main execution function for training and evaluation.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data
    # We use cached data if available to save time
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = BreastMILModel(pretrained=True)
    model.to(device)

    # 4. Optimization
    # BCEWithLogitsLoss is numerically stable
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop
    best_pf1 = -1.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_pf1 = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val pF1: {val_pf1}"
        )

        # Early Stopping & Checkpointing
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! (pF1: {best_pf1})")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Submission
    # Load best model
    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading best model from {Config.MODEL_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    predict_and_submit(model, test_loader, device)
