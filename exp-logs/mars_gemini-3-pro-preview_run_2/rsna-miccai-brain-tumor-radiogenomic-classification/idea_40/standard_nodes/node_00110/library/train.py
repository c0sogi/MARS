import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import DataCacher, MRIDataset, get_transforms
from library.model import AsymmetricEfficientNet

logger = get_logger(__name__)


def train_model():
    """
    Executes the training pipeline for the Asymmetric Grouped EfficientNet.

    Steps:
    1. Loads metadata and caches MRI data into RAM.
    2. Initializes the model, optimizer, and loss function.
    3. Runs the training loop with Stochastic Multi-Scale training.
    4. Validates using a Matched Ensemble (Stride 2 + Stride 5).
    5. Implements Early Stopping and saves the best model.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    logger.info(f"Starting training on device: {device}")

    # 2. Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA) or not os.path.exists(
        Config.VAL_METADATA
    ):
        logger.error(
            "Metadata files not found. Please ensure metadata generation is complete."
        )
        return

    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)

    # 3. Data Caching
    # We process data into RAM. DataCacher handles the disk caching logic.
    logger.info("Initializing Data Caching...")
    train_cache = DataCacher.process_data(
        df_train, cache_key="train", load_cached_data=True
    )
    val_cache = DataCacher.process_data(df_val, cache_key="val", load_cached_data=True)

    # 4. Datasets & Loaders
    # Train: Stochastic Stride (Random 2 or 5)
    train_dataset = MRIDataset(
        data_cache=train_cache,
        metadata_df=df_train,
        transform=get_transforms("train"),
        stride_mode="random",
    )

    # Validation: Matched Ensemble requires predictions from both Stride 2 and Stride 5
    val_dataset_s2 = MRIDataset(
        data_cache=val_cache,
        metadata_df=df_val,
        transform=get_transforms("val"),
        stride_mode=2,
    )
    val_dataset_s5 = MRIDataset(
        data_cache=val_cache,
        metadata_df=df_val,
        transform=get_transforms("val"),
        stride_mode=5,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation loaders (shuffle=False to ensure alignment for ensembling)
    val_loader_s2 = DataLoader(
        val_dataset_s2,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader_s5 = DataLoader(
        val_dataset_s5,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Model, Optimizer, Loss
    model = AsymmetricEfficientNet()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    logger.info(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # [B] -> [B, 1]

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0

        # --- Validation Phase (Matched Ensemble) ---
        model.eval()
        all_labels = []
        all_preds_s2 = []
        all_preds_s5 = []

        with torch.no_grad():
            # Pass 1: Stride 2
            for images, labels in val_loader_s2:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                all_preds_s2.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

            # Pass 2: Stride 5
            for images, _ in val_loader_s5:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                all_preds_s5.append(probs.cpu().numpy())

        # Concatenate batches
        y_true = np.concatenate(all_labels)
        y_pred_s2 = np.concatenate(all_preds_s2)
        y_pred_s5 = np.concatenate(all_preds_s5)

        # Ensemble Average
        y_pred_ensemble = (y_pred_s2 + y_pred_s5) / 2.0

        # Calculate Metric
        try:
            val_auc = roc_auc_score(y_true, y_pred_ensemble)
        except ValueError:
            val_auc = 0.5  # Handle edge case with single class in batch

        # Logging
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {avg_train_loss:.8f} - "
            f"Val AUC: {val_auc:.16f}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Val AUC: {best_auc:.16f}")
