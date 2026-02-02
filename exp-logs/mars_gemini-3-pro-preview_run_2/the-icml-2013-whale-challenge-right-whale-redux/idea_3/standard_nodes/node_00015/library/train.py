import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import get_loaders
from library.model import WhaleConvNeXt


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        # Reshape labels to match logits (B, 1)
        loss = criterion(logits, labels.unsqueeze(1))

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions and labels for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu()
        all_targets.append(labels.detach().cpu())
        all_preds.append(probs)

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

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
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels.unsqueeze(1))

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu()
            all_targets.append(labels.cpu())
            all_preds.append(probs)

    val_loss = running_loss / len(loader.dataset)

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    val_auc = calculate_roc_auc(all_targets, all_preds)

    return val_loss, val_auc


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)

    return np.concatenate(all_preds)


def run():
    """
    Main execution function for training and submission.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # get_loaders handles caching internally via preprocess_dataset
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    model = WhaleConvNeXt()
    model = model.to(device)

    # 4. Optimizer, Loss, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience = 5
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | Train AUC: {train_auc} | "
            f"Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 6. Submission Generation
    print("Generating submission...")

    # Load best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: Best model not found, using current model state.")

    # Predict
    predictions = predict_test(model, test_loader, device)

    # Flatten predictions from (N, 1) to (N,)
    predictions = predictions.flatten()

    # Prepare Submission DataFrame
    # Read test.csv to get the correct clip names
    test_df = pd.read_csv(Config.TEST_CSV)

    # Ensure lengths match
    if len(predictions) != len(test_df):
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match number of test files ({len(test_df)})."
        )

    submission = pd.DataFrame({"clip": test_df["clip"], "probability": predictions})

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
