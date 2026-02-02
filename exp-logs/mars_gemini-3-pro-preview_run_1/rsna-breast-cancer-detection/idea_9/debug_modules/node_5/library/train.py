import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import SpatialSymmetryDifferenceNet


def pf1_score(y_true, y_pred):
    """
    Calculates the Probabilistic F1 score (pF1).
    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)

    Where:
    pPrecision = pTP / (pTP + pFP)
    pRecall = pTP / (TP + FN)
    pTP = Sum(y_true * y_pred)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # pTP: Sum of probabilities for actual positives
    pTP = np.sum(y_true * y_pred)

    # Denominator for Precision: Sum(y_pred) (which is pTP + pFP)
    sum_preds = np.sum(y_pred)

    # Denominator for Recall: Sum(y_true) (Total actual positives, TP + FN)
    total_positives = np.sum(y_true)

    # Precision
    if sum_preds == 0:
        p_precision = 0.0
    else:
        p_precision = pTP / sum_preds

    # Recall
    if total_positives == 0:
        p_recall = 0.0
    else:
        p_recall = pTP / total_positives

    # F1
    if p_precision + p_recall == 0:
        pf1 = 0.0
    else:
        pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall)

    return pf1


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    total_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch_idx, (target_img, contra_img, labels) in enumerate(loader):
        target_img = target_img.to(device)
        contra_img = contra_img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass through Siamese Network
        logits = model(target_img, contra_img)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping is explicitly disabled as per strategy

        # Optimizer step
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / dataset_size
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set and computes pF1 score.
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for target_img, contra_img, labels in loader:
            target_img = target_img.to(device)
            contra_img = contra_img.to(device)
            labels = labels.to(device)

            logits = model(target_img, contra_img)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / dataset_size
    score = pf1_score(all_labels, all_probs)

    return avg_loss, score


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    Aggregates predictions by taking the max probability across views for each prediction_id.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for target_img, contra_img, prediction_ids in loader:
            target_img = target_img.to(device)
            contra_img = contra_img.to(device)

            logits = model(target_img, contra_img)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Collect results
            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    if not results:
        print("No predictions generated.")
        return

    df = pd.DataFrame(results)

    # Aggregate by prediction_id (Max probability across views)
    # This handles cases where multiple images map to the same prediction ID (e.g., CC and MLO views)
    df_agg = df.groupby("prediction_id", as_index=False)["cancer"].max()

    # Save submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_agg.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    epochs=Config.EPOCHS,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    load_cached_data=True,
):
    """
    Main driver function for the training pipeline.
    """
    # Reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data,
        debug=debug,
        debug_sample_size=debug_sample_size,
    )

    # Model Initialization
    model = SpatialSymmetryDifferenceNet()
    model.to(device)

    # Loss Function
    # Using weighted BCE to handle high class imbalance (1:47)
    pos_weight_val = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # Training Loop Variables
    best_score = -1.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val pF1: {val_score}")

        # Checkpointing & Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with pF1: {best_score}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best pF1: {best_score}")

    # Inference on Test Set
    print("Generating submission...")

    # Load best model weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model file found. Using current model state.")

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
