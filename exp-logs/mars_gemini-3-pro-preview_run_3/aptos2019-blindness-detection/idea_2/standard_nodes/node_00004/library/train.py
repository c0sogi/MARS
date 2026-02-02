import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_dataloaders
from library.model import EfficientNetRegressor


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape is (B, 1)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast(enabled=Config.USE_AMP):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        # Scaled Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    return running_loss / count


def validate_one_epoch(model, loader, device, loss_fn):
    """
    Evaluates the model on the validation set.
    Returns average loss and Quadratic Weighted Kappa score.
    """
    model.eval()
    running_loss = 0.0
    count = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            # Collect predictions and labels for metric calculation
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_loss = running_loss / count

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate QWK (utility handles rounding/clipping)
    qwk = quadratic_weighted_kappa(all_labels, all_preds)

    return val_loss, qwk


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.
    Returns processed integer predictions [0, 4].
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)
            all_preds.append(outputs.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    # Post-process regression outputs to ordinal labels
    preds_processed = np.clip(all_preds, 0, 4)
    preds_processed = np.round(preds_processed).astype(int)

    return preds_processed


def run(debug=Config.DEBUG, epochs=Config.EPOCHS):
    """
    Main execution function for training and submission generation.
    """
    seed_everything(Config.SEED)

    # Setup directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Device configuration
    device = torch.device(Config.DEVICE)

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # Initialize Model
    model = EfficientNetRegressor()
    model = model.to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=Config.USE_AMP)

    # Training Loop Variables
    best_qwk = -np.inf
    best_loss = np.inf
    patience_counter = 0
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, loss_fn
        )
        val_loss, val_qwk = validate_one_epoch(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val QWK: {val_qwk:.6f}"
        )

        # Save model with best QWK (Target Metric)
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), best_model_path)
            print(f"  New Best QWK: {best_qwk:.6f}. Model saved.")

        # Early Stopping based on Validation Loss (Optimization Objective)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val QWK: {best_qwk:.6f}")

    # ==========================================
    # Inference and Submission
    # ==========================================
    print("Running inference on test set...")

    # Load best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current model weights.")

    # Generate predictions
    predictions = predict_test(model, test_loader, device)

    # Create submission DataFrame
    test_df = test_loader.dataset.df
    submission = pd.DataFrame({"id_code": test_df["id_code"], "diagnosis": predictions})

    # Save submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return best_qwk
