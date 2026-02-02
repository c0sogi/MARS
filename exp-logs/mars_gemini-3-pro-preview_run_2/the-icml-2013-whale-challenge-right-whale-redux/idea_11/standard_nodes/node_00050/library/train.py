import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
)
from library.models import get_model
from library.dataset import WhaleDataset, load_dataset_data


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)
            targets.extend(labels.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    score = calculate_roc_auc(targets, preds)
    return losses.avg, score


def run_fold(fold_idx, model_name, train_loader, val_loader, device):
    """
    Orchestrates the training loop for a specific fold and model architecture.
    """
    print(f"Starting training for {model_name} - Fold {fold_idx}")

    # Initialize Model
    model = get_model(model_name, pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler (CosineAnnealingLR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop Variables
    best_score = 0.0
    patience_counter = 0
    save_filename = f"{model_name}_fold_{fold_idx}.pth"

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_score:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_score, save_filename
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Clean up
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return best_score


def train_all_folds():
    """
    Main driver for 5-Fold Stratified Cross-Validation training.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and Merge Data for CV
    train_data_split, train_labels_split, _ = load_dataset_data(
        Config.TRAIN_CSV, "train"
    )
    val_data_split, val_labels_split, _ = load_dataset_data(Config.VAL_CSV, "val")

    all_data = np.concatenate([train_data_split, val_data_split], axis=0)
    all_labels = np.concatenate([train_labels_split, val_labels_split], axis=0)

    print(f"Total training samples for CV: {len(all_data)}")

    # 2. Stratified K-Fold
    skf = StratifiedKFold(n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED)

    # 3. Iterate Models and Folds
    for model_name in Config.MODEL_NAMES:
        print(f"\n{'='*40}")
        print(f"Processing Architecture: {model_name}")
        print(f"{'='*40}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, all_labels)):
            print(f"\n--- Setup Fold {fold}/{Config.FOLDS} ---")

            # Split Data
            X_train, X_val = all_data[train_idx], all_data[val_idx]
            y_train, y_val = all_labels[train_idx], all_labels[val_idx]

            # Create Datasets
            train_dataset = WhaleDataset(X_train, y_train, is_training=True)
            val_dataset = WhaleDataset(X_val, y_val, is_training=False)

            # Weighted Sampler for Class Balance
            class_counts = np.bincount(y_train.astype(int))
            if len(class_counts) < 2:
                class_weights = [1.0, 1.0]
            else:
                class_weights = 1.0 / (class_counts + 1e-6)

            sample_weights = [class_weights[int(l)] for l in y_train]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )

            # Create Loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                sampler=sampler,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Run Training for this fold
            run_fold(fold, model_name, train_loader, val_loader, device)


def inference():
    """
    Generates predictions for the test set using all trained models (Ensemble).
    """
    print(f"\n{'='*40}")
    print("Running Inference & Ensembling")
    print(f"{'='*40}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    test_data, _, test_clips = load_dataset_data(Config.TEST_CSV, "test")
    test_dataset = WhaleDataset(test_data, None, is_training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Accumulate predictions
    final_preds = np.zeros(len(test_data))
    model_count = 0

    # Iterate over all trained models (Models x Folds)
    for model_name in Config.MODEL_NAMES:
        for fold in range(Config.FOLDS):
            filename = f"{model_name}_fold_{fold}.pth"
            filepath = os.path.join(Config.OUTPUT_DIR, filename)

            if not os.path.exists(filepath):
                print(f"Warning: Checkpoint {filename} not found. Skipping.")
                continue

            print(f"Ensembling: Loading {filename}...")

            # Initialize model and load weights
            model = get_model(model_name, pretrained=False)
            checkpoint = load_checkpoint(model, filename, device=device)
            model = model.to(device)
            model.eval()

            fold_preds = []

            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    fold_preds.extend(probs)

            final_preds += np.array(fold_preds)
            model_count += 1

            del model, checkpoint
            torch.cuda.empty_cache()

    if model_count == 0:
        print("Error: No models found for inference.")
        return

    # Average predictions (Soft Voting)
    final_preds /= model_count

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"clip": test_clips, "probability": final_preds})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    train_all_folds()
    inference()


if __name__ == "__main__":
    main()
