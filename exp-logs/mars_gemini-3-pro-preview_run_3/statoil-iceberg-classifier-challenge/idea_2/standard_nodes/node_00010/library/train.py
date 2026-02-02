import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    N_FOLDS,
    SEED,
    WORKING_DIR,
    SUBMISSION_FILE,
)
from library.utils import seed_everything, save_checkpoint
from library.data_loader import (
    load_and_process_data,
    get_fold_loaders,
    get_test_loader,
)
from library.model import (
    IcebergVGG16,
    train_one_epoch,
    validate,
    predict,
)


def run_fold(
    fold_idx,
    X_train,
    y_train,
    angles_train,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    patience=EARLY_STOPPING_PATIENCE,
    device=DEVICE,
):
    """
    Manages the training loop for a single cross-validation fold.
    """
    print(f"\n{'='*20}")
    print(f"Starting Fold {fold_idx + 1} / {N_FOLDS}")
    print(f"{'='*20}")

    # 1. Prepare DataLoaders
    train_loader, val_loader = get_fold_loaders(
        fold_idx, X_train, y_train, angles_train, batch_size=batch_size
    )

    # 2. Initialize Model
    model = IcebergVGG16(dropout_rate=0.5).to(device)

    # 3. Setup Optimizer and Loss
    # We optimize only the classifier parameters as the backbone is frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    fold_checkpoint_dir = os.path.join(WORKING_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)

        # Validate
        val_loss = validate(val_loader, model, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss}"
        )

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                checkpoint_dir=fold_checkpoint_dir,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return best_val_loss


def train_and_predict(load_cached=True):
    """
    Main pipeline:
    1. Loads data.
    2. Runs 5-fold CV training.
    3. Generates ensemble predictions.
    4. Saves submission.
    """
    seed_everything(SEED)
    print(f"Running training pipeline on device: {DEVICE}")

    # 1. Load Data
    # Uses caching mechanism implemented in data_loader
    X_train, y_train, angles_train, X_test, ids_test, angles_test = (
        load_and_process_data(load_cached_data=load_cached)
    )

    # 2. Train Folds
    for fold in range(N_FOLDS):
        run_fold(
            fold,
            X_train,
            y_train,
            angles_train,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            patience=EARLY_STOPPING_PATIENCE,
            device=DEVICE,
        )

    # 3. Inference and Ensembling
    print("\nStarting Inference...")
    test_loader = get_test_loader(X_test, angles_test, batch_size=BATCH_SIZE)

    # Accumulate predictions
    test_preds_accum = np.zeros(len(X_test))

    for fold in range(N_FOLDS):
        print(f"Predicting with model from Fold {fold + 1}...")

        # Load best model for this fold
        fold_checkpoint_dir = os.path.join(WORKING_DIR, f"fold_{fold}")
        best_model_path = os.path.join(fold_checkpoint_dir, "model_best.pth")

        model = IcebergVGG16(dropout_rate=0.5).to(DEVICE)
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

        # Predict
        fold_preds = predict(test_loader, model, DEVICE)
        test_preds_accum += fold_preds

    # Average predictions
    avg_preds = test_preds_accum / N_FOLDS

    # 4. Save Submission
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": avg_preds})
    df_sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved successfully to: {SUBMISSION_FILE}")
