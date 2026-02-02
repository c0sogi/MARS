import sys
import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_loader import get_dataloaders, process_data
from library.model import (
    ParallelFactorizedDCNResNet,
    train_one_epoch,
    validate,
    predict_test,
)

# Initialize Logger
logger = get_logger()


def perform_failure_analysis(model, device):
    """
    Analyzes model failure modes by correlating feature values with error magnitude
    on the validation set.
    """
    logger.info("Starting Failure Analysis...")

    # Load raw validation data (numpy arrays)
    # process_data returns: train_X, train_y, val_X, val_y, test_X, test_ids
    _, _, val_X, val_y, _, _ = process_data(load_cached_data=True)

    # Prepare for inference
    model.eval()
    batch_size = Config.BATCH_SIZE
    num_samples = val_X.shape[0]
    error_magnitudes = []

    # Calculate Error Magnitude: 1.0 - Probability(True Class)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # Prepare batch
            batch_X_np = val_X[i : i + batch_size]
            batch_y_np = val_y[i : i + batch_size]

            batch_X = torch.FloatTensor(batch_X_np).to(device)
            # Convert targets from 1-7 range to 0-6 index for gathering
            batch_y_idx = torch.LongTensor(batch_y_np - 1).to(device)

            # Forward pass
            logits = model(batch_X)
            probs = torch.softmax(logits, dim=1)

            # Get probability assigned to the true class
            # gather requires index tensor to have same dims as probs
            true_class_probs = probs.gather(1, batch_y_idx.unsqueeze(1)).squeeze(1)

            # Error = 1 - p(true)
            batch_errors = 1.0 - true_class_probs.cpu().numpy()
            error_magnitudes.append(batch_errors)

    all_errors = np.concatenate(error_magnitudes)

    # Calculate Correlation between each feature and the error magnitude
    correlations = []
    num_features = val_X.shape[1]

    for col_idx in range(num_features):
        feature_vals = val_X[:, col_idx]

        # Avoid warning/NaN if feature is constant
        if np.std(feature_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_vals, all_errors)[0, 1]

        correlations.append((col_idx, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("-" * 40)
    print("Failure Analysis: Top Correlated Features with Error")
    print("-" * 40)
    for i in range(min(10, len(correlations))):
        idx, corr = correlations[i]
        print(f"Feature Index {idx}: Correlation = {corr:.6f}")
    print("-" * 40)


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    logger.info("Generating submission...")
    ids, preds = predict_test(model, test_loader, device)

    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    df_sub = pd.DataFrame({"Id": ids, "Cover_Type": preds})

    # Ensure strict integer types
    df_sub["Id"] = df_sub["Id"].astype(int)
    df_sub["Cover_Type"] = df_sub["Cover_Type"].astype(int)

    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved successfully to {Config.SUBMISSION_FILE}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Determine dimensions
    dummy_X, _ = next(iter(train_loader))
    input_dim = dummy_X.shape[1]
    num_classes = 7

    # 3. Model Initialization
    model = ParallelFactorizedDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        dcn_rank=Config.DCN_RANK,
        hidden_dim=Config.HIDDEN_DIM,
        resnet_blocks=Config.RESNET_BLOCKS,
        dropout=Config.DROPOUT,
    ).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_ETA_MIN
    )

    # 5. Training Loop
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        # Train & Validate
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging
        logger.info(
            f"Epoch {epoch}/{Config.EPOCHS} | LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f} | "
            f"Time: {time.time() - epoch_start:.2f}s"
        )

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    logger.info(
        f"Training finished in {total_time:.2f}s. Best Val Acc: {best_val_acc:.8f}"
    )

    # 6. Load Best Model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 7. Final Validation Metric (Required Output)
    # Re-run validation on the best model to ensure accuracy
    final_val_loss, final_val_acc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_val_acc:.20f}")

    # 8. Failure Analysis
    perform_failure_analysis(model, device)

    # 9. Conditional Submission
    # Threshold: 0.9625041666666667
    SUBMISSION_THRESHOLD = 0.9625041666666667

    if final_val_acc > SUBMISSION_THRESHOLD:
        logger.info(
            f"Metric {final_val_acc:.6f} > {SUBMISSION_THRESHOLD:.6f}. Generating submission."
        )
        generate_submission(model, test_loader, device)
    else:
        logger.info(
            f"Metric {final_val_acc:.6f} <= {SUBMISSION_THRESHOLD:.6f}. Submission skipped."
        )


if __name__ == "__main__":
    main()
