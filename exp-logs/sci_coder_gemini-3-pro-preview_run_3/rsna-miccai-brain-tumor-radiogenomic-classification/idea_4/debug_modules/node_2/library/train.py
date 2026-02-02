import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import Siamese25DNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch_idx, (flair, t1w, t1wce, t2w, targets) in enumerate(loader):
        flair = flair.to(device)
        t1w = t1w.to(device)
        t1wce = t1wce.to(device)
        t2w = t2w.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        logits = model(flair, t1w, t1wce, t2w)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        count += targets.size(0)

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (flair, t1w, t1wce, t2w, targets) in enumerate(loader):
            flair = flair.to(device)
            t1w = t1w.to(device)
            t1wce = t1wce.to(device)
            t2w = t2w.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(flair, t1w, t1wce, t2w)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * targets.size(0)
            count += targets.size(0)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / count if count > 0 else 0.0

    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Handle edge case where only one class is present in validation set (unlikely with stratification)
        if len(np.unique(all_targets)) > 1:
            auc_score = roc_auc_score(all_targets, all_probs)
        else:
            auc_score = 0.5
    else:
        auc_score = 0.5

    return epoch_loss, auc_score


def predict_and_submit(
    model, loader, device, output_path="./submission/submission.csv"
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    ids_list = []
    probs_list = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for batch_idx, (flair, t1w, t1wce, t2w, ids) in enumerate(loader):
            flair = flair.to(device)
            t1w = t1w.to(device)
            t1wce = t1wce.to(device)
            t2w = t2w.to(device)

            logits = model(flair, t1w, t1wce, t2w)
            probs = torch.sigmoid(logits)

            ids_list.extend(ids)
            probs_list.extend(probs.cpu().numpy().flatten())

    # Create submission directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Format BraTS21ID as 5-digit string for consistency, though sample uses int
    # The sample submission uses int for IDs (e.g. 356), so we keep them as is or convert to int.
    # Based on sample_submission.csv provided in description:
    # BraTS21ID,MGMT_value
    # 00001,0.5
    # But description also says "BraTS21ID (int64)".
    # The utils.py process_patient converts ID to string.
    # We will format it to match the sample submission requirement if needed,
    # but usually keeping it as the ID provided in the loader is safest.
    # The loader returns IDs as strings (from utils.py ids_list).

    submission_df = pd.DataFrame({"BraTS21ID": ids_list, "MGMT_value": probs_list})

    # Ensure BraTS21ID is formatted correctly (5 digits)
    # The sample submission in description shows "00001", "00013".
    # The loader loads them as strings.

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    epochs=20,
    batch_size=8,
    learning_rate=1e-4,
    patience=5,
    debug=False,
    load_cached_data=True,
    save_dir="./working/idea_4",
):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(42)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}, LR: {learning_rate}, Epochs: {epochs}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=2,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 3. Model Initialization
    # We use 32 input channels as defined in the idea description
    model = Siamese25DNet(
        backbone_name="efficientnet_b0", pretrained=True, in_channels=32, num_classes=1
    )
    model = model.to(device)

    # 4. Optimization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    predict_and_submit(
        model, test_loader, device, output_path="./submission/submission.csv"
    )

    return best_auc
