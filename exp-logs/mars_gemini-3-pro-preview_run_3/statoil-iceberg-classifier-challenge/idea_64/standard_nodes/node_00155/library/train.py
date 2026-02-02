import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.utils import set_seed, logger, train_one_epoch, validate
from library.data import get_fold_loaders, get_test_loader
from library.model import MSICNN


def run_training(epochs=75, batch_size=32, patience=12, load_cached_data=True):
    """
    Orchestrates the 5-fold cross-validation training and submission generation.

    Args:
        epochs (int): Maximum number of training epochs per fold.
        batch_size (int): Batch size for dataloaders.
        patience (int): Early stopping patience.
        load_cached_data (bool): Whether to use cached numpy arrays for data.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Retrieve test IDs and initialize prediction accumulator
    # This call also ensures data is processed and cached if not already done
    _, test_ids = get_test_loader(
        batch_size=batch_size, load_cached_data=load_cached_data
    )
    test_preds_accum = np.zeros(len(test_ids))

    n_splits = 5

    for fold in range(n_splits):
        logger.info(f"Starting Fold {fold+1}/{n_splits}")

        # Get DataLoaders for this fold
        # This function handles the leak-free median angle calculation
        train_loader, val_loader, fold_median = get_fold_loaders(
            fold_index=fold,
            n_splits=n_splits,
            batch_size=batch_size,
            load_cached_data=load_cached_data,
            seed=42,
        )

        # Initialize Model
        model = MSICNN().to(device)

        # Optimizer and Loss
        # Using constants as defined in the solution design
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop variables
        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # train_one_epoch handles the multi-sample dropout loss aggregation
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            # Print full precision metrics
            print(
                f"Fold {fold+1} Epoch {epoch+1} Train Loss: {train_loss} Val Loss: {val_loss}"
            )

            # Checkpoint and Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Inference on Test Set with Best Model
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # Get Test Loader using the median angle from the current training fold
        test_loader, _ = get_test_loader(
            batch_size=batch_size,
            load_cached_data=load_cached_data,
            angle_impute_val=fold_median,
        )

        fold_preds = []
        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Forward pass returns logits in eval mode
                logits = model(images, angles)
                probs = torch.sigmoid(logits)
                fold_preds.append(probs.cpu().numpy())

        # Accumulate predictions
        fold_preds_flat = np.concatenate(fold_preds).flatten()
        test_preds_accum += fold_preds_flat

        # Cleanup to free GPU memory
        del model, optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache()

    # Average predictions across all folds
    avg_preds = test_preds_accum / n_splits

    # Save Submission
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})
    submission.to_csv(submission_path, index=False)
    logger.info(f"Submission saved successfully to {submission_path}")
