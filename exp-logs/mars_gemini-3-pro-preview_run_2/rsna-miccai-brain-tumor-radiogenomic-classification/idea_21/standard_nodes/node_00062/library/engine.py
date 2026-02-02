import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, get_logger, AverageMeter
from library.dataset import BraTSDataset
from library.model import AsymmetricEfficientNet


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, logger):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    # Use tqdm for progress tracking if running interactively, otherwise silent
    # based on logger usage, but prompt asks not to print progress bars.
    # We will iterate silently or with simple logging.

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass (logits)
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, criterion, device, logger):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(outputs)

            losses.update(loss.item(), images.size(0))
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Handle edge case where only one class is present in batch (rare in full val)
    # Cite debug_lesson_1
    if len(np.unique(all_targets)) < 2:
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc = 0.5

    return losses.avg, auc


def predict_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    Strategies: Original, Horizontal Flip, Vertical Flip.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, subject_ids in loader:
            images = images.to(device)

            # 1. Original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip (axis 3: [B, C, H, W])
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip (axis 2: [B, C, H, W])
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v)
            probs_v = torch.sigmoid(logits_v)

            # Average probabilities
            avg_probs = (probs_orig + probs_h + probs_v) / 3.0

            # Store results
            avg_probs_np = avg_probs.cpu().numpy().flatten()
            subject_ids_np = subject_ids.numpy().flatten()

            for sid, prob in zip(subject_ids_np, avg_probs_np):
                results.append({"BraTS21ID": sid, "MGMT_value": prob})

    return pd.DataFrame(results)


def run_training(logger=None):
    """
    Main driver function to run the training pipeline with Early Stopping.
    """
    if logger is None:
        logger = get_logger()

    seed_everything(Config.SEED)

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    logger.info(f"Training samples: {len(df_train)}")
    logger.info(f"Validation samples: {len(df_val)}")

    # 2. Create Datasets & Loaders
    train_dataset = BraTSDataset(df_train, phase="train", load_cached_data=True)
    val_dataset = BraTSDataset(df_val, phase="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Setup
    device = torch.device(Config.DEVICE)
    model = AsymmetricEfficientNet()
    model = model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, logger
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device, logger)

        logger.info(
            f"Epoch {epoch}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val AUC: {val_auc:.15f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            logger.info(f"New best model saved with AUC: {best_auc:.15f}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    return best_auc


def generate_submission(logger=None):
    """
    Loads the best model and generates predictions for the test set using TTA.
    """
    if logger is None:
        logger = get_logger()

    seed_everything(Config.SEED)

    # 1. Load Metadata
    df_test = pd.read_csv(Config.TEST_CSV)
    logger.info(f"Test samples: {len(df_test)}")

    # 2. Dataset & Loader
    test_dataset = BraTSDataset(df_test, phase="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    device = torch.device(Config.DEVICE)
    model = AsymmetricEfficientNet()

    if os.path.exists(Config.BEST_MODEL_PATH):
        logger.info(f"Loading model from {Config.BEST_MODEL_PATH}")
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            "Best model not found! Using random initialization (this should not happen in valid runs)."
        )

    model = model.to(device)

    # 4. Inference with TTA
    logger.info("Running inference with TTA...")
    df_preds = predict_tta(model, test_loader, device)

    # 5. Format Submission
    # Ensure BraTS21ID is 5 digits in the output if needed, but sample submission uses int.
    # The sample submission shows: 00001,0.5 -> actually the csv content provided in prompt shows:
    # BraTS21ID,MGMT_value
    # 00001,0.5
    # But the sample_submission.csv description says BraTS21ID is int64.
    # We will stick to the format BraTS21ID (5-digit string) if the sample requires it,
    # OR just match the sample_submission.csv provided in the prompt.
    # Prompt sample:
    # BraTS21ID,MGMT_value
    # 00001,0.5
    # This implies 5-digit zero padding.

    # However, the prompt also says:
    # "The file should contain a header and have the following format:
    # BraTS21ID,MGMT_value
    # 00001,0.5"

    # Let's ensure the ID is formatted correctly.
    df_preds["BraTS21ID"] = df_preds["BraTS21ID"].apply(lambda x: f"{int(x):05d}")

    # Sort by ID
    df_preds = df_preds.sort_values("BraTS21ID")

    # Save
    df_preds.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    return df_preds
