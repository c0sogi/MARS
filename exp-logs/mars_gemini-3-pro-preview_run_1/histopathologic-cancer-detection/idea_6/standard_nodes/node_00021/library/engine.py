import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, calculate_auc
from library.models import get_model
from library.dataset import get_fold_dataloaders, get_test_dataloader


def train_one_epoch(model, loader, optimizer, device, criterion, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for _, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    if scheduler is not None:
        scheduler.step()

    return losses.avg


def evaluate(model, loader, device, criterion):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            losses.update(loss.item(), images.size(0))
            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

    auc = calculate_auc(all_targets, all_preds)
    return losses.avg, auc


def train_fold(fold_idx, model_name):
    """
    Orchestrates the training process for a single fold.
    Handles model initialization, training loop, early stopping, and saving.

    Args:
        fold_idx (int): The fold index (0-4).
        model_name (str): The architecture name ('densenet121' or 'densenet169').

    Returns:
        tuple: (path_to_best_model, best_auc_score)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Model
    model = get_model(model_name, pretrained=Config.PRETRAINED)
    model = model.to(device)

    # 2. Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 3. Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # 4. Data Loaders
    train_loader, val_loader = get_fold_dataloaders(fold_idx)

    # 5. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(
        Config.WORK_DIR, f"{model_name}_fold{fold_idx}_best.pth"
    )

    print(f"[{model_name}] Starting training for Fold {fold_idx} on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, criterion, scheduler
        )
        val_loss, val_auc = evaluate(model, val_loader, device, criterion)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Save Best Model (maximize AUC)
        if val_auc > best_auc + Config.MIN_DELTA:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"[{model_name}] Fold {fold_idx} finished. Best AUC: {best_auc:.10f}")
    return best_model_path, best_auc


def predict_submission(models_config):
    """
    Generates the final submission file using the trained ensemble.
    Applies Test Time Augmentation (TTA).

    Args:
        models_config (list): List of tuples (model_name, checkpoint_path).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_test_dataloader()

    # 1. Load all models in the ensemble
    models = []
    print(f"Loading {len(models_config)} models for inference...")
    for name, path in models_config:
        m = get_model(name, pretrained=False)
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        m.eval()
        models.append(m)

    results = {}
    print(f"Starting inference with {Config.TTA_VIEWS} TTA views per image...")

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # 2. Generate TTA Views
            # View 1: Original
            views = [images]
            # View 2: Horizontal Flip
            views.append(torch.flip(images, dims=[3]))
            # View 3: Vertical Flip
            views.append(torch.flip(images, dims=[2]))
            # View 4: Rotate 90 degrees
            views.append(torch.rot90(images, k=1, dims=[2, 3]))

            batch_probs = []

            # 3. Predict with every model on every view
            for model in models:
                for view in views:
                    logits = model(view)
                    probs = torch.sigmoid(logits)
                    batch_probs.append(probs.cpu().numpy())

            # 4. Average Predictions
            # Stack shape: (num_models * num_views, batch_size, 1)
            stacked_probs = np.stack(batch_probs, axis=0)
            avg_probs = np.mean(stacked_probs, axis=0)  # (batch_size, 1)

            # 5. Store Results
            for i, img_id in enumerate(ids):
                results[img_id] = avg_probs[i][0]

    # 6. Save Submission
    df = pd.DataFrame({"id": list(results.keys()), "label": list(results.values())})

    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
