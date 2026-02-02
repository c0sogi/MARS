import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import BirdDataset, get_transforms
from library.model import get_model
from library.utils import set_seed, calculate_multilabel_auc, save_submission


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates loss for mixed inputs.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(
    model, loader, optimizer, criterion, device, epoch, scheduler=None, mixup_alpha=0.0
):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Apply Mixup if enabled
        if mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, mixup_alpha, device
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)

    # Step the scheduler if it is an epoch-level scheduler (like CosineAnnealingLR)
    if scheduler is not None:
        scheduler.step()

    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    auc_score = calculate_multilabel_auc(all_targets, all_preds)

    return total_loss, auc_score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns tuple of (predictions, rec_ids).
    """
    model.eval()
    all_preds = []
    all_rec_ids = []

    with torch.no_grad():
        for images, _, rec_ids in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_rec_ids.extend(rec_ids.numpy())

    return np.vstack(all_preds), np.array(all_rec_ids)


def run_training(
    max_samples=Config.MAX_SAMPLES,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
    mixup_alpha=0.4,
):
    """
    Main function to run the training pipeline.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # ==========================================
    # 1. Data Preparation
    # ==========================================
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")

    train_dataset = BirdDataset(
        metadata_path=Config.TRAIN_CSV,
        mode="train",
        transform=train_transform,
        max_samples=max_samples,
    )
    val_dataset = BirdDataset(
        metadata_path=Config.VAL_CSV,
        mode="val",
        transform=val_transform,
        max_samples=max_samples,
    )
    test_dataset = BirdDataset(
        metadata_path=Config.TEST_CSV,
        mode="test",
        transform=val_transform,
        max_samples=max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # ==========================================
    # 2. Model Initialization
    # ==========================================
    model = get_model(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ==========================================
    # 3. Training Loop with Early Stopping
    # ==========================================
    best_auc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            scheduler,
            mixup_alpha,
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val AUC:    {val_auc:.10f}")

        # Early Stopping Logic
        if val_auc > best_auc:
            print(
                f"  [Improvement] AUC increased from {best_auc:.10f} to {val_auc:.10f}. Saving model..."
            )
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            # Save checkpoint to disk
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            epochs_no_improve += 1
            print(f"  [No Improvement] Patience: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val AUC: {best_auc:.10f}")

    # ==========================================
    # 4. Inference and Submission
    # ==========================================
    print("Loading best model for inference...")
    model.load_state_dict(best_model_wts)

    print("Generating predictions on Test set...")
    predictions, test_rec_ids = predict(model, test_loader, device)

    print(f"Saving submission to {Config.PREDICTIONS_PATH}...")
    save_submission(predictions, test_rec_ids, Config.PREDICTIONS_PATH)

    print("Done.")
