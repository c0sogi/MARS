import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, MetricMonitor
from library.dataset import get_dataloaders
from library.models import WhaleEfficientNet, WhaleDenseNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    return metric_monitor


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set and computes ROC-AUC.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    preds = []
    targets = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device).float().unsqueeze(1)

            output = model(data)
            loss = criterion(output, target)

            metric_monitor.update("Loss", loss.item())

            # Sigmoid for probability
            prob = torch.sigmoid(output)
            preds.extend(prob.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())

    try:
        auc = roc_auc_score(targets, preds)
    except ValueError:
        # Fallback if only one class is present in the batch (rare in validation)
        auc = 0.5

    metric_monitor.update("AUC", auc)
    return metric_monitor, auc


def train_model_instance(
    model_class, model_name, save_path, train_loader, val_loader, device
):
    """
    Instantiates and trains a specific model architecture with Early Stopping.
    """
    print(f"\n[{model_name}] Initializing...")
    model = model_class().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = getattr(optim, Config.OPTIMIZER)(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = getattr(optim.lr_scheduler, Config.SCHEDULER)(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LR
    )

    best_auc = 0.0
    patience_counter = 0

    print(f"[{model_name}] Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print full precision metrics
        print(f"Epoch {epoch} | {train_metrics} | {val_metrics}")

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"--> Best Model Saved! AUC: {best_auc}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print(f"[{model_name}] Early stopping triggered.")
            break

    # Load best weights before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    return model


def generate_predictions(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_clips = []
    all_probs = []

    with torch.no_grad():
        for data, clips in loader:
            data = data.to(device)
            output = model(data)
            probs = torch.sigmoid(output).cpu().numpy().flatten()

            all_clips.extend(clips)
            all_probs.extend(probs)

    return np.array(all_clips), np.array(all_probs)


def train():
    """
    Main execution function to train the model and generate submission.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 2. Train Model (EfficientNet)
    model = train_model_instance(
        WhaleEfficientNet,
        "EfficientNet-B0",
        Config.MODEL_PATH,
        train_loader,
        val_loader,
        device,
    )

    # 3. Inference
    print("\nRunning Inference...")
    clips, probs = generate_predictions(model, test_loader, device)

    # 4. Save Submission
    submission = pd.DataFrame({"clip": clips, "probability": probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())
