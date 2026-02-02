import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from sklearn.metrics import f1_score
from scipy import stats

# Add current directory to sys.path
sys.path.append(os.getcwd())

# Import provided library modules
from library.config import Config
from library.utils import (
    set_seed,
    get_logger,
    load_checkpoint,
    calculate_metrics,
    EarlyStopping,
)
from library.data import get_dataloaders
from library.model import DualStreamTextCNN
from library.train import Trainer
from library.inference import predict_probs, optimize_threshold, generate_submission


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for Fast Baseline
    Config.DEBUG = True
    Config.NUM_EPOCHS = 5
    Config.BATCH_SIZE = 128  # Smaller batch size for the smaller debug dataset

    # Create working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Logger
    logger = get_logger("runfile")
    logger.info("Starting Fast Baseline Execution...")

    # Set Seed & Device
    set_seed(Config.SEED)
    device = Config.get_device()
    logger.info(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    logger.info("Loading Data...")
    # load_cached_data=True allows using pre-processed data if available
    train_loader, val_loader, test_loader, vocab, mlb = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    num_classes = len(mlb.classes_)
    logger.info(f"Data loaded. Vocab size: {len(vocab)}, Num classes: {num_classes}")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = DualStreamTextCNN(num_classes=num_classes)
    model.to(device)

    # ==========================================
    # 4. Training Setup
    # ==========================================
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: One Cycle Policy
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.NUM_EPOCHS,
        pct_start=0.1,
    )

    scaler = GradScaler()

    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=True, path=Config.MODEL_SAVE_PATH
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        early_stopping=early_stopping,
    )

    # ==========================================
    # 5. Training Loop
    # ==========================================
    logger.info("Starting Training...")
    trainer.fit(Config.NUM_EPOCHS)

    # ==========================================
    # 6. Evaluation
    # ==========================================
    logger.info("Loading best model for evaluation...")
    try:
        start_epoch, best_metric = load_checkpoint(
            Config.MODEL_SAVE_PATH, model, device=device
        )
        logger.info(f"Loaded checkpoint from epoch {start_epoch-1}")
    except FileNotFoundError:
        logger.error("No checkpoint found. Training might have failed.")
        return

    model.eval()

    # Inference on Validation Set
    logger.info("Running inference on Validation set...")
    val_probs = predict_probs(model, val_loader, device)

    # Get Ground Truth (Sparse)
    val_targets_sparse = val_loader.dataset.labels

    # Optimize Threshold
    best_threshold = optimize_threshold(val_probs, val_targets_sparse)

    # Calculate Final Metric
    val_preds = val_probs > best_threshold
    final_metric = f1_score(
        val_targets_sparse, val_preds, average="samples", zero_division=0
    )

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 7. Failure Analysis
    # ==========================================
    logger.info("Performing Failure Analysis...")

    # Convert sparse targets to dense for element-wise analysis
    # Note: In DEBUG mode this is small. In full mode, chunking would be preferred.
    val_targets_dense = val_targets_sparse.toarray()

    # Compute F1 per sample manually
    # TP: Both pred and target are 1
    tp = np.sum(val_preds & (val_targets_dense == 1), axis=1)
    # FP: Pred is 1, target is 0
    fp = np.sum(val_preds & (val_targets_dense == 0), axis=1)
    # FN: Pred is 0, target is 1
    fn = np.sum((~val_preds) & (val_targets_dense == 1), axis=1)

    epsilon = 1e-7
    f1_per_sample = 2 * tp / (2 * tp + fp + fn + epsilon)

    # Error Magnitude (1 - F1)
    error_magnitude = 1.0 - f1_per_sample

    # Get Input Features (Lengths)
    # Count non-padding tokens
    title_lens = np.sum(val_loader.dataset.title_ids != 0, axis=1)
    body_lens = np.sum(val_loader.dataset.body_ids != 0, axis=1)

    # Calculate Pearson Correlation
    corr_title, _ = stats.pearsonr(error_magnitude, title_lens)
    corr_body, _ = stats.pearsonr(error_magnitude, body_lens)

    print(f"Correlation between Error and Title Length: {corr_title}")
    print(f"Correlation between Error and Body Length: {corr_body}")

    # ==========================================
    # 8. Submission
    # ==========================================
    TARGET_METRIC = 0.33488

    if final_metric > TARGET_METRIC:
        logger.info(
            f"Validation Metric ({final_metric}) > Target ({TARGET_METRIC}). Generating Submission..."
        )

        # Inference on Test Set
        test_probs = predict_probs(model, test_loader, device)
        test_ids = test_loader.dataset.ids

        # Generate CSV
        generate_submission(
            test_probs, test_ids, best_threshold, mlb, Config.SUBMISSION_FILE
        )
    else:
        logger.info(
            f"Validation Metric ({final_metric}) <= Target ({TARGET_METRIC}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
