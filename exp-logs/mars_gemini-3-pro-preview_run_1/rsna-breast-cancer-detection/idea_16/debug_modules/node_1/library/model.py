import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger, set_seed, probabilistic_f1
from library.data import get_dataloaders
from library.modules import SiameseFPNModel

# Initialize logger for this module
logger = get_logger("model")


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    # Progress bar for feedback
    pbar = tqdm(
        loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]", leave=False
    )

    for batch in pbar:
        # Unpack batch: target_img, contra_img, label
        # Shapes: [B, 3, H, W], [B, 3, H, W], [B]
        target_img, contra_img, labels = batch

        target_img = target_img.to(device)
        contra_img = contra_img.to(device)
        labels = labels.to(device).unsqueeze(1)  # [B, 1]

        optimizer.zero_grad()

        # Forward pass
        logits = model(target_img, contra_img)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping (if enabled in Config)
        if Config.CLIP_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Probabilistic F1 score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="[Val]", leave=False)
        for batch in pbar:
            target_img, contra_img, labels = batch

            target_img = target_img.to(device)
            contra_img = contra_img.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(target_img, contra_img)
            loss = criterion(logits, labels)

            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    # Calculate pF1
    pf1 = probabilistic_f1(all_labels, all_preds)

    return avg_loss, pf1


def predict_and_submit(model, loader, device):
    """
    Runs inference on the test set and generates the submission file.
    Aggregates predictions by taking the max probability per prediction_id.
    """
    logger.info("Starting inference on test set...")
    model.eval()

    results = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="[Test]", leave=False)
        for batch in pbar:
            target_img, contra_img, prediction_ids = batch

            target_img = target_img.to(device)
            contra_img = contra_img.to(device)

            logits = model(target_img, contra_img)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Store (prediction_id, probability)
            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Create DataFrame
    df_results = pd.DataFrame(results)

    if df_results.empty:
        logger.warning(
            "No predictions generated. Creating empty submission with defaults."
        )
        # Fallback for empty test set (though unlikely)
        df_sub = pd.DataFrame(columns=["prediction_id", "cancer"])
    else:
        # Aggregate: Group by prediction_id and take MAX probability
        # This handles cases where multiple views (CC, MLO) map to the same breast ID
        df_sub = df_results.groupby("prediction_id", as_index=False)["cancer"].max()

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(
        f"Submission saved to {Config.SUBMISSION_PATH} with {len(df_sub)} rows."
    )


def run():
    """
    Main execution function.
    Handles setup, training loop, validation, and submission.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data
    # load_cached_data=True allows skipping processing if cache exists
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    logger.info(f"Initializing model: {Config.PROJECT_NAME}")
    model = SiameseFPNModel()
    model.to(device)

    # 4. Optimizer & Loss
    # pos_weight handles class imbalance (approx 47.0)
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_pf1 = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    logger.info("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        logger.info(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val pF1: {val_pf1}"
        )  # Full precision as requested

        # Checkpoint (Save Best)
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved! (pF1: {best_pf1})")

    # 6. Inference
    logger.info("Training complete. Loading best model for inference...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logger.warning("No best model found. Using current model state.")

    predict_and_submit(model, test_loader, device)
    logger.info("Run complete.")
