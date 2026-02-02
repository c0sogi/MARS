import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import (
    tqdm,
)  # Not used for printing, but good practice to have if needed, suppressed here.

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import FractureMILModel
from library.loss import HierarchicalCompoundLoss

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (volumes, targets) in enumerate(loader):
        volumes = volumes.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model returns logits for C1-C7 (Batch, 7)
        logits = model(volumes)

        # Loss calculation
        # HierarchicalCompoundLoss expects logits (B, 7) and targets (B, 8)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * volumes.size(0)
        dataset_size += volumes.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for volumes, targets in loader:
            volumes = volumes.to(device)
            targets = targets.to(device)

            logits = model(volumes)
            loss = criterion(logits, targets)

            running_loss += loss.item() * volumes.size(0)
            dataset_size += volumes.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def inference(model, loader, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    model.eval()
    results = []

    # Subtypes as defined in the competition
    subtypes = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

    with torch.no_grad():
        for i, (volumes, _) in enumerate(loader):
            volumes = volumes.to(device)

            # Get study UIDs for this batch
            # The loader returns (volume, label), but we need StudyInstanceUID to map rows.
            # Since the dataset class doesn't return UIDs in __getitem__, we rely on
            # the order being preserved. The test_loader is not shuffled.
            # We access the dataframe from the dataset directly.
            start_idx = i * loader.batch_size
            end_idx = start_idx + volumes.size(0)
            batch_uids = loader.dataset.df.iloc[start_idx:end_idx][
                "StudyInstanceUID"
            ].values

            # Forward pass
            logits = model(volumes)  # (B, 7)

            # Apply sigmoid to get probabilities for C1-C7
            probs_c1_c7 = torch.sigmoid(logits)  # (B, 7)

            # Derive patient_overall probability (Max of C1-C7)
            probs_patient, _ = torch.max(probs_c1_c7, dim=1, keepdim=True)  # (B, 1)

            # Concatenate: (B, 8) -> [C1, ..., C7, patient_overall]
            all_probs = torch.cat([probs_c1_c7, probs_patient], dim=1).cpu().numpy()

            # Create rows for submission
            for j, study_uid in enumerate(batch_uids):
                study_probs = all_probs[j]
                for k, subtype in enumerate(subtypes):
                    row_id = f"{study_uid}_{subtype}"
                    prob = study_probs[k]
                    results.append({"row_id": row_id, "fractured": prob})

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Save to file
    submission_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")


def run(debug=False):
    """
    Main driver function to train, validate, and infer.
    """
    seed_everything(Config.SEED)

    # Debugging setting
    sample_size = Config.DEBUG_SAMPLE_SIZE if debug else None

    # 1. Data Loading
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug_sample_size=sample_size
    )

    # 2. Model Initialization
    device = torch.device(Config.DEVICE)
    model = FractureMILModel().to(device)

    # 3. Training Setup
    criterion = HierarchicalCompoundLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    # T_max is based on total epochs * multiplier
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(Config.EPOCHS * Config.T_MAX_MULT), eta_min=Config.MIN_LR
    )

    # Early Stopping parameters
    patience = 3
    patience_counter = 0
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    logger.info(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    # 4. Training Loop
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with Val Loss: {val_loss}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # 5. Inference
    logger.info("Loading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logger.warning("Best model file not found. Using current model state.")

    logger.info("Generating predictions on test set...")
    inference(model, test_loader, device)

    logger.info("Run completed.")
