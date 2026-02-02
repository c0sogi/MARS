import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.cuda.amp import GradScaler

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import SMTSINModel


def pf1_score(labels, preds):
    """
    Computes the Probabilistic F1 score (pF1).
    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)

    Where:
    pPrecision = pTP / (pTP + pFP)
    pRecall = pTP / (TP + FN)
    pTP = Sum of probabilities for positive class instances
    pFP = Sum of probabilities for negative class instances
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    # Masks
    pos_mask = labels == 1
    neg_mask = labels == 0

    # Probabilistic True Positives (Sum of probs for positive class)
    pTP = preds[pos_mask].sum()

    # Probabilistic False Positives (Sum of probs for negative class)
    pFP = preds[neg_mask].sum()

    # Total Actual Positives (TP + FN)
    total_positives = pos_mask.sum()

    # Precision
    if (pTP + pFP) == 0:
        p_precision = 0.0
    else:
        p_precision = pTP / (pTP + pFP)

    # Recall
    if total_positives == 0:
        p_recall = 0.0
    else:
        p_recall = pTP / total_positives

    # F1
    if (p_precision + p_recall) == 0:
        pf1 = 0.0
    else:
        pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall)

    return pf1


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, epoch):
    """
    Trains the model for one epoch.
    Handles mixed precision for forward pass and Float32 for loss calculation.
    """
    model.train()
    running_loss = 0.0

    # Define Loss Functions
    # Primary: Weighted BCE for Cancer
    pos_weight = torch.tensor([Config.POS_WEIGHT], device=device)
    criterion_cancer = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Auxiliary: MSE for BIRADS, CrossEntropy for Density
    criterion_birads = nn.MSELoss()
    criterion_density = nn.CrossEntropyLoss()

    for batch_idx, (images, metadata, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)

        # Prepare targets
        t_cancer = targets["cancer"].to(device, non_blocking=True).view(-1, 1)
        t_birads = targets["birads"].to(device, non_blocking=True).view(-1, 1)
        t_density = targets["density"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
            outputs = model(images, metadata)

        # Float32 Loss Calculation
        # We explicitly disable autocast and cast outputs to float32 to prevent
        # numerical instability caused by high pos_weight in BCE loss.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # Cast outputs to float32
            o_cancer = outputs["cancer"].float()
            o_birads = outputs["birads"].float()
            o_density = outputs["density"].float()

            # 1. Primary Loss (Cancer)
            loss_c = criterion_cancer(o_cancer, t_cancer)

            # 2. Aux Loss 1 (BIRADS) - Mask out missing values (-1)
            mask_b = (t_birads != -1).flatten()
            if mask_b.sum() > 0:
                loss_b = criterion_birads(o_birads[mask_b], t_birads[mask_b])
            else:
                loss_b = torch.tensor(0.0, device=device)

            # 3. Aux Loss 2 (Density) - Mask out missing values (-1)
            mask_d = t_density != -1
            if mask_d.sum() > 0:
                loss_d = criterion_density(o_density[mask_d], t_density[mask_d])
            else:
                loss_d = torch.tensor(0.0, device=device)

            # Weighted Sum
            loss = (
                Config.LOSS_WEIGHT_CANCER * loss_c
                + Config.LOSS_WEIGHT_BIRADS * loss_b
                + Config.LOSS_WEIGHT_DENSITY * loss_d
            )

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Probabilistic F1 score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, metadata, targets in loader:
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            t_cancer = targets["cancer"].float().numpy()

            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                outputs = model(images, metadata)

            # Extract cancer probabilities (Sigmoid)
            probs = torch.sigmoid(outputs["cancer"]).float().cpu().numpy()

            all_preds.append(probs)
            all_targets.append(t_cancer)

    # Concatenate results
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # Compute Metric
    score = pf1_score(all_targets, all_preds)
    return score


def inference(model, loader, device):
    """
    Runs inference on the test set and generates the submission file.
    Aggregates predictions by prediction_id using Max Pooling.
    """
    model.eval()
    results = []

    print("Running inference on Test set...")
    with torch.no_grad():
        for images, metadata, pred_ids in loader:
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                outputs = model(images, metadata)

            # Get probabilities
            probs = torch.sigmoid(outputs["cancer"]).float().cpu().numpy().flatten()

            # Store results
            for pid, prob in zip(pred_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Create DataFrame
    df = pd.DataFrame(results)

    # Aggregation: Max probability per prediction_id
    # This aligns with the Single-Instance Learning strategy where we take the most suspicious view
    submission = df.groupby("prediction_id")["cancer"].max().reset_index()

    # Save Submission
    Config.create_dirs()
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(debug=Config.DEBUG):
    """
    Main execution function for training and inference.
    """
    seed_everything(Config.SEED)
    device = get_device()

    print(f"Device: {device}")
    print(f"Debug Mode: {debug}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=debug
    )

    # 2. Initialize Model
    model = SMTSINModel().to(device)

    # 3. Setup Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # Scaler for Mixed Precision
    scaler = GradScaler(enabled=Config.USE_AMP)

    # 4. Training Loop
    best_pf1 = 0.0
    patience = 3  # Early stopping patience
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, epoch
        )
        val_pf1 = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val pF1: {val_pf1:.6f}"
        )

        # Save Best Model
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            patience_counter = 0
            print(f"New best model saved with pF1: {best_pf1:.6f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished. Best Val pF1: {best_pf1:.6f}")

    # 5. Inference
    # Load best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: No best model found. Using current model.")

    inference(model, test_loader, device)
