import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

from library.config import (
    TRAIN_JSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_JSON,
    TEST_META_PATH,
    CHECKPOINT_DIR,
    SUBMISSION_FILE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    NUM_WORKERS,
    SEED,
    DEVICE,
    N_FOLDS,
)
from library.utils import set_seed, print_metrics, generate_submission_file
from library.data_loader import load_data_split, IcebergDataset, seed_worker
from library.model import DPDB_HSE_CNN


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, epoch, total_epochs
):
    """
    Trains the model for one epoch.
    Updates DropBlock probability based on training progress.
    """
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # Linearly increase DropBlock probability
    # Progress goes from 0.0 to 1.0 over the course of training
    progress = epoch / total_epochs
    if hasattr(model, "update_dropblock_prob"):
        model.update_dropblock_prob(progress)

    for images, angles, labels, _ in dataloader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Calculate accuracy for monitoring
        preds = torch.sigmoid(outputs) > 0.5
        correct_preds += (preds == (labels > 0.5)).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds / total_preds

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, angles, labels, _ in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs) > 0.5
            correct_preds += (preds == (labels > 0.5)).sum().item()
            total_preds += labels.size(0)

    val_loss = running_loss / total_preds
    val_acc = correct_preds / total_preds

    return val_loss, val_acc


def run_fold(fold_idx, train_loader, val_loader, device):
    """
    Runs training for a single fold.
    """
    print(f"\n--- Starting Fold {fold_idx} ---")

    model = DPDB_HSE_CNN().to(device)

    # AdamW with constant learning rate
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # BCEWithLogitsLoss combines Sigmoid and BCELoss
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss


def predict_test(model, test_loader, device):
    """
    Generates predictions for the test set using a single model.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, angles, _, _ in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy().flatten())

    return np.array(preds)


def train_and_evaluate():
    """
    Main function to run 5-Fold CV and generate submission.
    """
    set_seed(SEED)

    # 1. Load Data
    # We load both train and val splits from metadata and merge them to perform full 5-Fold CV
    print("Loading data for Cross-Validation...")
    X_train_part, ang_train_part, y_train_part, id_train_part = load_data_split(
        TRAIN_META_PATH, TRAIN_JSON, "train", load_cached_data=True
    )
    X_val_part, ang_val_part, y_val_part, id_val_part = load_data_split(
        VAL_META_PATH, TRAIN_JSON, "val", load_cached_data=True
    )

    # Merge
    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    ang_full = np.concatenate([ang_train_part, ang_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)
    id_full = np.concatenate([id_train_part, id_val_part], axis=0)

    print(f"Total training samples: {len(y_full)}")

    # Load Test Data
    X_test, ang_test, _, id_test = load_data_split(
        TEST_META_PATH, TEST_JSON, "test", load_cached_data=True
    )

    # Impute Test Angles globally (using full train median)
    # Note: Inside CV, we impute train/val based on fold, but for test we use global train stats
    global_angle_median = np.nanmedian(ang_full)
    ang_test_imputed = np.where(np.isnan(ang_test), global_angle_median, ang_test)

    # 2. Prepare Cross-Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )
    val_transform = None

    # Generator for reproducible DataLoaders
    g = torch.Generator()
    g.manual_seed(SEED)

    fold_metrics = []
    test_predictions_sum = np.zeros(len(X_test))

    # 3. Run Folds
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Split Data
        X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
        ang_train_fold, ang_val_fold = ang_full[train_idx], ang_full[val_idx]
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]
        id_train_fold, id_val_fold = id_full[train_idx], id_full[val_idx]

        # Impute Incidence Angles (Median of training fold)
        fold_median = np.nanmedian(ang_train_fold)
        ang_train_fold = np.where(np.isnan(ang_train_fold), fold_median, ang_train_fold)
        ang_val_fold = np.where(np.isnan(ang_val_fold), fold_median, ang_val_fold)

        # Create Datasets
        train_dataset = IcebergDataset(
            X_train_fold,
            ang_train_fold,
            y_train_fold,
            id_train_fold,
            transform=train_transform,
        )
        val_dataset = IcebergDataset(
            X_val_fold, ang_val_fold, y_val_fold, id_val_fold, transform=val_transform
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Train Fold
        best_loss = run_fold(fold_idx, train_loader, val_loader, DEVICE)
        fold_metrics.append(best_loss)

        # 4. Inference on Test Set for this Fold
        # Load best model
        model = DPDB_HSE_CNN().to(DEVICE)
        model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        # Create Test Loader
        test_dataset = IcebergDataset(
            X_test, ang_test_imputed, None, id_test, transform=val_transform
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        print(f"Generating predictions for Fold {fold_idx}...")
        fold_preds = predict_test(model, test_loader, DEVICE)
        test_predictions_sum += fold_preds

    # 5. Average Predictions and Save Submission
    avg_preds = test_predictions_sum / N_FOLDS

    print("\n--- Cross-Validation Summary ---")
    for i, loss in enumerate(fold_metrics):
        print(f"Fold {i}: {loss:.6f}")
    print(f"Average Val Loss: {np.mean(fold_metrics):.6f}")

    print(f"Saving ensemble submission to {SUBMISSION_FILE}...")
    generate_submission_file(avg_preds, id_test, SUBMISSION_FILE)
