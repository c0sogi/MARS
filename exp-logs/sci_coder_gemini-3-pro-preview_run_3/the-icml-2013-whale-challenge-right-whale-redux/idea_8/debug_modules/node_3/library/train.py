import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, get_score, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model import WhaleDetector
from library.loss import WeightedBCELoss


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup augmentation and weighted loss mixing.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Apply Mixup if enabled and batch size is sufficient
        if Config.MIXUP_ALPHA > 0 and batch_size > 1:
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]
            y_a, y_b = labels, labels[index]

            outputs = model(mixed_images)

            # Compute loss by mixing the weighted scalar losses of the pairs
            # This preserves the class weighting mechanism better than mixing labels
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets)

    val_loss = running_loss / dataset_size

    # Concatenate results
    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()

    val_auc = get_score(all_targets, all_preds)

    return val_loss, val_auc


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)

    all_preds = np.concatenate(all_preds).ravel()
    return all_preds


def run_training():
    """
    Main execution function:
    1. Sets up data, model, optimizer.
    2. Runs training loop with early stopping.
    3. Reloads best model.
    4. Generates submission file.
    """
    seed_everything(Config.SEED)

    # 1. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # 2. Model Setup
    device = torch.device(Config.DEVICE)
    print(f"Initializing Model on {device}...")
    model = WhaleDetector(pretrained=Config.PRETRAINED)
    model.to(device)

    # 3. Optimization Setup
    criterion = WeightedBCELoss(device=device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 4. Training Loop
    best_score = -np.inf
    patience = 5
    counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{Config.EPOCHS} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss: {val_loss}")
        print(f"  Val AUC: {val_auc}")

        # Save Best Model
        if val_auc > best_score:
            print(f"  Score improved from {best_score} to {val_auc}. Saving model...")
            best_score = val_auc
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_auc, Config.BEST_MODEL_PATH
            )
            counter = 0
        else:
            counter += 1
            print(f"  Score did not improve. Patience: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("\nStarting inference on test set...")

    if Config.RELOAD_BEST_MODEL and os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        load_checkpoint(model, Config.BEST_MODEL_PATH, device=device)
    else:
        print("Using current model weights for inference.")

    preds = inference(model, test_loader, device)

    # 6. Submission Generation
    print("Generating submission file...")
    test_df = pd.read_csv(Config.TEST_CSV)

    # Adjust test_df if running in debug mode (dataloader subsets data)
    if Config.DEBUG:
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

    # Safety check for length mismatch
    if len(preds) != len(test_df):
        print(
            f"Warning: Prediction count ({len(preds)}) does not match Metadata count ({len(test_df)}). Truncating to match."
        )
        min_len = min(len(preds), len(test_df))
        preds = preds[:min_len]
        test_df = test_df.iloc[:min_len]

    submission = pd.DataFrame({"clip": test_df["clip_name"], "probability": preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
