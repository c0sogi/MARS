import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, AverageMeter, kl_divergence_score
from library.data import get_dataloaders
from library.models import BidirectionalFusionNet


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = AverageMeter()
    kl_meter = AverageMeter()

    for batch in loader:
        # Move data to device
        eeg = batch["eeg"].to(device)
        spec = batch["spec"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(eeg, spec)

        # Loss calculation: KLDivLoss expects log-probabilities as input
        log_probs = F.log_softmax(logits, dim=1)
        loss = nn.KLDivLoss(reduction="batchmean")(log_probs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Metrics
        # Calculate KL score using the utility function for consistency
        # Apply softmax to get probabilities for the metric function
        probs = F.softmax(logits, dim=1)
        kl_score = kl_divergence_score(targets, probs)

        loss_meter.update(loss.item(), eeg.size(0))
        kl_meter.update(kl_score, eeg.size(0))

    print(f"Epoch [{epoch}] Train Loss: {loss_meter.avg} | Train KL: {kl_meter.avg}")
    return loss_meter.avg


def validate(model, loader, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()
    kl_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            spec = batch["spec"].to(device)
            targets = batch["target"].to(device)

            logits = model(eeg, spec)

            # Loss
            log_probs = F.log_softmax(logits, dim=1)
            loss = nn.KLDivLoss(reduction="batchmean")(log_probs, targets)

            # Metric
            probs = F.softmax(logits, dim=1)
            kl_score = kl_divergence_score(targets, probs)

            loss_meter.update(loss.item(), eeg.size(0))
            kl_meter.update(kl_score, eeg.size(0))

    print(f"Validation Loss: {loss_meter.avg}")
    print(f"Validation KL: {kl_meter.avg}")
    return loss_meter.avg


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    predictions = []
    eeg_ids = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch["eeg"].to(device)
            spec = batch["spec"].to(device)
            batch_eeg_ids = batch["eeg_id"]

            logits = model(eeg, spec)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            predictions.append(probs)
            eeg_ids.extend(batch_eeg_ids.numpy())

    predictions = np.concatenate(predictions, axis=0)

    # Create DataFrame
    df_sub = pd.DataFrame({"eeg_id": eeg_ids})

    # Add probability columns
    # Column names must match sample_submission format
    # The config defines CLASS_NAMES as ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
    # The submission requires columns like "seizure_vote", etc.
    vote_cols = [f"{c}_vote" for c in Config.CLASS_NAMES]

    for i, col in enumerate(vote_cols):
        df_sub[col] = predictions[:, i]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train_model():
    """
    Main function to orchestrate training, validation, and submission.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(Config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model
    model = BidirectionalFusionNet(Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 5. Training Loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{Config.EPOCHS} ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss = validate(model, val_loader, device)

        # Checkpoint & Early Stopping
        if val_loss < best_loss - Config.MIN_DELTA:
            print(
                f"Validation loss improved from {best_loss} to {val_loss}. Saving model..."
            )
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Submission
    if test_loader is not None:
        # Load best model
        print("Loading best model for inference...")
        if os.path.exists(Config.BEST_MODEL_PATH):
            model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=device)
            )
        else:
            print("Warning: Best model not found, using current model state.")

        generate_submission(model, test_loader, device, Config.SUBMISSION_CSV)
    else:
        print("No test data found. Skipping submission generation.")
