import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import AsymmetricEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device, label_smoothing=0.0):
    """
    Executes one epoch of training.
    Manually implements label smoothing for BCEWithLogitsLoss.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Apply Label Smoothing: new_y = y * (1 - alpha) + 0.5 * alpha
        if label_smoothing > 0:
            targets = labels * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            targets = labels

        optimizer.zero_grad()
        # Model returns [Batch, 1], squeeze to [Batch]
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    count = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / count if count > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate AUC only if there are valid labels (sanity check)
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = 0.5
    else:
        auc = 0.5

    return epoch_loss, auc


def predict_and_submit(model, test_loader, device, output_path):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    Saves the result to submission.csv.
    TTA Strategy: Average of (Original, Horizontal Flip, Vertical Flip).
    """
    model.eval()
    results = []

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # 1. Original
            out_orig = model(images).squeeze(1)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (dim 3: [B, C, H, W])
            images_h = torch.flip(images, [3])
            out_h = model(images_h).squeeze(1)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (dim 2)
            images_v = torch.flip(images, [2])
            out_v = model(images_v).squeeze(1)
            prob_v = torch.sigmoid(out_v)

            # Average probabilities
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0

            # Map back to BraTS21ID
            # Calculate indices in the original dataframe
            start_idx = i * test_loader.batch_size
            end_idx = start_idx + images.size(0)

            # Access the underlying dataframe to get IDs
            # Note: test_loader.dataset is an MRIDataset instance
            batch_ids = test_loader.dataset.df.iloc[start_idx:end_idx][
                "BraTS21ID"
            ].values

            for bid, prob in zip(batch_ids, avg_prob.cpu().numpy()):
                results.append({"BraTS21ID": bid, "MGMT_value": prob})

    df_sub = pd.DataFrame(results)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    train_loader=None,
    val_loader=None,
    test_loader=None,
    epochs=None,
    load_cached_data=True,
):
    """
    Main driver function for the training pipeline.
    Handles data loading, model initialization, training loop, early stopping, and submission.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data Loading
    if train_loader is None or val_loader is None or test_loader is None:
        train_loader, val_loader, test_loader = get_dataloaders(
            load_cached_data=load_cached_data
        )

    # 2. Model Initialization
    model = AsymmetricEfficientNet()
    model = model.to(device)

    # 3. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    num_epochs = epochs if epochs is not None else Config.EPOCHS
    patience = Config.PATIENCE
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            label_smoothing=Config.LABEL_SMOOTHING,
        )

        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")

    # 5. Inference & Submission
    # Load the best model weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)

    return model
