import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import configuration and utilities from the provided library
from library.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    N_FOLDS,
    SEED,
    NUM_WORKERS,
    DEVICE,
    IDEA_DIR,
    PATIENCE,
    MIN_DELTA,
    SUBMISSION_PATH,
)
from library.utils import (
    seed_everything,
    AverageMeter,
    get_logger,
    print_full_precision_metrics,
    save_checkpoint,
)
from library.dataset import BraTSDataset, get_transforms
from library.model import CASIVNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    losses = AverageMeter("Train Loss")

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (Batch, 1)

        optimizer.zero_grad()

        # Forward pass (logits)
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Validation loop. Returns average loss and ROC AUC.
    """
    model.eval()
    losses = AverageMeter("Val Loss")

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(logits)

            losses.update(loss.item(), images.size(0))

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Handle edge case where only one class is present in the batch/subset
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return losses.avg, auc


def predict_test_set(models, device):
    """
    Generates predictions for the test set using an ensemble of fold models.
    """
    test_ds = BraTSDataset(split="test", transform=get_transforms("test"))
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Dictionary to store accumulated probabilities: {braTS21ID: sum_prob}
    # We use a dictionary because the loader returns IDs
    results = {}
    counts = {}

    print(f"Running inference on test set with {len(models)} models...")

    for fold_idx, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                ids = ids.numpy()

                for i, pid in enumerate(ids):
                    if pid not in results:
                        results[pid] = 0.0
                        counts[pid] = 0
                    results[pid] += probs[i]
                    counts[pid] += 1

    # Average predictions
    final_preds = []
    for pid in sorted(results.keys()):
        avg_prob = results[pid] / counts[pid]
        final_preds.append({"BraTS21ID": pid, "MGMT_value": avg_prob})

    return pd.DataFrame(final_preds)


def run_kfold_training():
    """
    Main execution function for 5-Fold Cross-Validation training and submission generation.
    """
    seed_everything(SEED)

    # Setup logging
    log_path = os.path.join(IDEA_DIR, "training.log")
    logger = get_logger(log_path)
    logger.info(f"Starting training with device: {DEVICE}")
    logger.info(
        f"Hyperparameters: BS={BATCH_SIZE}, LR={LEARNING_RATE}, WD={WEIGHT_DECAY}"
    )

    # Load Datasets
    # We use the training split from metadata for CV
    full_train_ds = BraTSDataset(split="train", transform=get_transforms("train"))
    # We need a copy with validation transforms (no augmentation) for the validation phase
    full_val_ds = BraTSDataset(split="train", transform=get_transforms("val"))

    # Prepare K-Fold
    # We use the labels from the dataset for stratified splitting
    labels = full_train_ds.labels
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels)
    ):
        logger.info(f"\n{'='*20} Fold {fold} {'='*20}")

        # Create Subsets
        train_subset = Subset(full_train_ds, train_idx)
        val_subset = Subset(full_val_ds, val_idx)

        # Create Loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = CASIVNet()
        model.to(DEVICE)

        # Optimizer & Loss
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop Variables
        best_auc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(IDEA_DIR, f"best_model_fold{fold}.pth")

        for epoch in range(EPOCHS):
            start_time = time.time()

            # Train
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )

            # Validate
            val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

            elapsed = time.time() - start_time

            # Log metrics
            metrics = {
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Val AUC": val_auc,
                "Time": f"{elapsed:.2f}s",
            }
            print_full_precision_metrics(
                metrics, phase=f"Fold {fold} - Epoch {epoch+1}"
            )

            # Checkpointing & Early Stopping
            if val_auc > best_auc + MIN_DELTA:
                best_auc = val_auc
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_auc": best_auc,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=True,
                    save_dir=IDEA_DIR,
                    filename=f"checkpoint_fold{fold}.pth",
                )
                # Rename the generic best_model.pth created by save_checkpoint to fold specific
                if os.path.exists(os.path.join(IDEA_DIR, "best_model.pth")):
                    os.rename(os.path.join(IDEA_DIR, "best_model.pth"), best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model for this fold to use in inference
        logger.info(f"Fold {fold} finished. Best AUC: {best_auc}")

        # Re-load the best state to ensure the model list has the optimized weights
        best_checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(best_checkpoint["state_dict"])
        fold_models.append(model)

    # ==========================================
    # Submission Generation
    # ==========================================
    logger.info("\nGenerating submission...")

    submission_df = predict_test_set(fold_models, DEVICE)

    # Save submission
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {SUBMISSION_PATH}")

    # Also save to working dir for easy access/verification
    working_sub_path = os.path.join(IDEA_DIR, "submission.csv")
    submission_df.to_csv(working_sub_path, index=False)
