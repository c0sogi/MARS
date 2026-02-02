import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import seed_everything, MetricMonitor, get_checkpoint_path
from library.dataset import get_dataloaders, get_test_loader
from library.model import WhaleEnsembleMember


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device).unsqueeze(1)  # Shape (Batch, 1)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    return metric_monitor


def validate(model, loader, criterion, device):
    """
    Validates the model and calculates ROC AUC.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device).unsqueeze(1)

            logits = model(data)
            loss = criterion(logits, target)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            metric_monitor.update("Loss", loss.item())

            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Handle edge case where batch might only have one class (though unlikely with stratified split)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return metric_monitor, auc


def train_individual_model(model_name, train_loader, val_loader):
    """
    Instantiates and trains a specific model architecture.
    Returns the path to the best checkpoint.
    """
    print(f"\n=== Training Model: {model_name} ===")

    device = Config.DEVICE
    model = WhaleEnsembleMember(model_name, pretrained=Config.PRETRAINED)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    best_auc = 0.0
    patience_counter = 0
    best_model_path = get_checkpoint_path(model_name)

    for epoch in range(1, Config.EPOCHS + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch} | "
            f"Train {train_metrics} | "
            f"Val {val_metrics} | "
            f"Val AUC: {val_auc}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Best AUC Improved. Saved model to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    return best_model_path


def inference(model, loader, device):
    """
    Generates probabilities for the test set.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().flatten())

    return np.array(all_probs)


def run_training():
    """
    Main execution function.
    1. Loads data.
    2. Trains each model in the ensemble.
    3. Generates predictions.
    4. Averages predictions (Soft Voting).
    5. Saves submission.
    """
    seed_everything(Config.SEED)

    # Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        load_cached_data=True, debug=Config.DEBUG
    )
    test_loader, test_clips = get_test_loader(load_cached_data=True, debug=Config.DEBUG)

    ensemble_predictions = []

    # Iterate through defined models in the ensemble
    for model_name in Config.MODEL_NAMES:
        # Train
        best_ckpt_path = train_individual_model(model_name, train_loader, val_loader)

        # Load Best for Inference
        print(f"Loading best checkpoint for {model_name} from {best_ckpt_path}...")
        model = WhaleEnsembleMember(
            model_name, pretrained=False
        )  # Pretrained weights not needed when loading state_dict
        model.load_state_dict(torch.load(best_ckpt_path, map_location=Config.DEVICE))
        model = model.to(Config.DEVICE)

        # Predict
        print(f"Generating predictions for {model_name}...")
        preds = inference(model, test_loader, Config.DEVICE)
        ensemble_predictions.append(preds)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Ensemble: Soft Voting (Average Probabilities)
    print("\nAggregating Ensemble Predictions...")
    ensemble_predictions = np.array(ensemble_predictions)
    avg_predictions = np.mean(ensemble_predictions, axis=0)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"clip": test_clips, "probability": avg_predictions})

    # Save Submission
    print(f"Saving submission to {Config.OUTPUT_SUBMISSION_PATH}...")
    submission_df.to_csv(Config.OUTPUT_SUBMISSION_PATH, index=False)
    print("Done.")
