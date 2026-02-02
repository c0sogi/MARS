import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, calculate_metric
from library.data import get_dataloaders
from library.model import SLHDAN


def build_baseline_lookup(dataset):
    """
    Constructs a lookup dictionary for Baseline FVC and Baseline Week
    from the dataset's internal dataframe. This is necessary because the
    training __getitem__ does not return baseline anchors, which are
    required for the parametric trajectory calculation in the model.

    Args:
        dataset: LungDataset instance

    Returns:
        dict: {patient_id: {'base_fvc': float, 'base_week': float}}
    """
    df = dataset.df
    # The dataframe in dataset already has Baseline_FVC and Baseline_Weeks
    # merged by get_dataloaders logic.
    # We drop duplicates to get unique patient entries.
    unique_df = df[["Patient", "Baseline_FVC", "Baseline_Weeks"]].drop_duplicates(
        subset=["Patient"]
    )

    lookup = {}
    for _, row in unique_df.iterrows():
        lookup[row["Patient"]] = {
            "base_fvc": float(row["Baseline_FVC"]),
            "base_week": float(row["Baseline_Weeks"]),
        }
    return lookup


def get_baseline_tensors(patient_ids, lookup, device):
    """
    Retrieves baseline values for a batch of patients using the lookup dict.

    Args:
        patient_ids (list): List of patient IDs from the batch.
        lookup (dict): The lookup dictionary.
        device (torch.device): Device to place tensors on.

    Returns:
        tuple: (base_fvc_tensor, base_week_tensor) of shape (B, 1)
    """
    base_fvcs = []
    base_weeks = []

    for pid in patient_ids:
        vals = lookup[pid]
        base_fvcs.append(vals["base_fvc"])
        base_weeks.append(vals["base_week"])

    # Shape (B, 1)
    t_fvc = torch.tensor(base_fvcs, dtype=torch.float32, device=device).unsqueeze(1)
    t_week = torch.tensor(base_weeks, dtype=torch.float32, device=device).unsqueeze(1)

    return t_fvc, t_week


def train_one_epoch(model, loader, optimizer, criterion, device, lookup):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch
        img_ax = batch["image_ax"].to(device)
        img_cor = batch["image_cor"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device)  # FVC
        current_weeks = batch["weeks"].to(device).unsqueeze(1)  # (B, 1)
        patient_ids = batch["patient_id"]

        # Get baseline anchors from lookup to enable trajectory calculation
        base_fvc, base_week = get_baseline_tensors(patient_ids, lookup, device)

        optimizer.zero_grad()

        # Forward pass with temporal args to get Trajectory Predictions
        # Output: (B, 2) -> [Pred_FVC, Pred_Sigma]
        preds = model(
            img_ax,
            img_cor,
            tabular,
            base_fvc=base_fvc,
            base_week=base_week,
            current_week=current_weeks,
        )

        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_ax.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device, lookup):
    """
    Evaluates the model on the validation set.
    Returns the average competition metric (higher is better).
    """
    model.eval()
    running_metric = 0.0

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["image_ax"].to(device)
            img_cor = batch["image_cor"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)
            current_weeks = batch["weeks"].to(device).unsqueeze(1)
            patient_ids = batch["patient_id"]

            base_fvc, base_week = get_baseline_tensors(patient_ids, lookup, device)

            preds = model(
                img_ax,
                img_cor,
                tabular,
                base_fvc=base_fvc,
                base_week=base_week,
                current_week=current_weeks,
            )

            # Calculate Metric (Negative Loss)
            # calculate_metric returns the competition score (higher is better)
            metric = calculate_metric(preds, targets)
            running_metric += metric * img_ax.size(0)

    return running_metric / len(loader.dataset)


def generate_submission(model, loader, device):
    """
    Generates predictions for the test set and creates the submission dataframe.
    """
    model.eval()

    # We use the loader's dataframe to get the Patient_Week identifiers
    # The loader preserves order (shuffle=False), so we can map 1:1.
    test_df = loader.dataset.df

    predictions = []
    confidences = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["image_ax"].to(device)
            img_cor = batch["image_cor"].to(device)
            tabular = batch["tabular"].to(device)

            # Test loader explicitly provides these keys (see library/data.py)
            base_fvc = batch["base_fvc"].to(device).view(-1, 1)
            base_week = batch["base_week"].to(device).view(-1, 1)
            predict_week = batch["predict_week"].to(device).view(-1, 1)

            preds = model(
                img_ax,
                img_cor,
                tabular,
                base_fvc=base_fvc,
                base_week=base_week,
                current_week=predict_week,
            )

            # preds is (B, 2) -> [FVC, Sigma]
            batch_fvc = preds[:, 0].cpu().numpy()
            batch_sigma = preds[:, 1].cpu().numpy()

            predictions.extend(batch_fvc)
            confidences.extend(batch_sigma)

    # Create submission dataframe
    sub_df = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": predictions,
            "Confidence": confidences,
        }
    )

    return sub_df


def run_training():
    """
    Main execution function:
    1. Sets up data, model, optimizer.
    2. Runs training loop with early stopping.
    3. Generates submission file.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing SLH-DAN on {device}...")

    # 1. Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Build lookups for Train/Val baseline anchors
    # This bridges the gap between the dataset structure and model requirements
    train_lookup = build_baseline_lookup(train_loader.dataset)
    val_lookup = build_baseline_lookup(val_loader.dataset)

    # 2. Model
    model = SLHDAN().to(device)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLogLikelihoodLoss().to(device)

    # 4. Training Loop
    best_metric = -float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, train_lookup
        )

        # Validate
        val_metric = validate(model, val_loader, criterion, device, val_lookup)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  -> New Best Model Saved! (Metric: {val_metric:.6f})")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    print("Generating submission...")
    sub_df = generate_submission(model, test_loader, device)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(sub_df.head())
