import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed
from library.model import MGMTNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape: (B, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Input images shape: (B, C, H, W)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    # Use BCELoss for validation tracking
    val_criterion = nn.BCELoss()

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)  # Shape: (B, 1)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Calculate loss
            loss = val_criterion(probs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    total_loss = running_loss / dataset_size

    # Calculate ROC AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    return total_loss, auc_score


def predict_test_set(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, _, subject_ids in dataloader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Collect results
            for i in range(len(subject_ids)):
                results.append(
                    {"BraTS21ID": subject_ids[i].item(), "MGMT_value": probs[i][0]}
                )

    return pd.DataFrame(results)


def run(train_loader, val_loader, test_loader):
    """
    Main driver function to train, evaluate, and generate submission.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Model Initialization
    model = MGMTNet(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        in_chans=Config.NUM_CHANNELS,
        drop_rate=Config.DROPOUT_RATE,
    )
    model = model.to(device)

    # 3. Optimization
    # BCEWithLogitsLoss is more stable than BCELoss + Sigmoid
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 5. Inference & Submission
    print("Loading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model file not found. Using current model state.")

    print("Generating predictions on test set...")
    df_submission = predict_test_set(model, test_loader, device)

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
