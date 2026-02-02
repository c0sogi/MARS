import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            # Handle scalar output for batch size 1
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)

            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Original, H-Flip, V-Flip).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, subject_ids in loader:
            images = images.to(device)

            # 1. Original
            out_orig = torch.sigmoid(model(images).squeeze())

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            out_h = torch.sigmoid(model(images_h).squeeze())

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            out_v = torch.sigmoid(model(images_v).squeeze())

            # Average
            avg_preds = (out_orig + out_h + out_v) / 3.0

            # Handle scalar output
            if avg_preds.ndim == 0:
                avg_preds = avg_preds.unsqueeze(0)

            for i, pid in enumerate(subject_ids):
                results.append(
                    {"BraTS21ID": pid.item(), "MGMT_value": avg_preds[i].item()}
                )

    return results


def run_training(
    epochs=10,
    batch_size=32,
    debug_limit=None,
    save_path="./working/best_model.pth",
    patience=5,
):
    """
    Main training loop with early stopping.
    """
    seed_everything(42)
    device = get_device()

    # Ensure working directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, debug_limit=debug_limit, load_cached_data=True
    )

    # Model Setup
    model = AsymmetricEfficientNet(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            # print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")
    return best_auc


def generate_submission(
    model_path="./working/best_model.pth",
    output_path="./submission/submission.csv",
    batch_size=32,
):
    """
    Generates submission file for the test set.
    """
    seed_everything(42)
    device = get_device()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get Test Loader
    _, _, test_loader = get_dataloaders(batch_size=batch_size, load_cached_data=True)

    # Load Model
    model = AsymmetricEfficientNet(pretrained=False).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model path {model_path} not found. Using random weights.")

    # Predict
    print("Generating predictions with TTA...")
    results = predict_with_tta(model, test_loader, device)

    # Save
    df = pd.DataFrame(results)
    df = df[["BraTS21ID", "MGMT_value"]]
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
