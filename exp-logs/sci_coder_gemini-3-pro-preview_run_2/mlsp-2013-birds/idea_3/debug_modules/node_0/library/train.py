import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import load_data, BirdDataset, get_transforms
from library.model import BirdClassifier
from library.engine import train_one_epoch, validate_one_epoch


def run_kfold_training(
    debug=Config.DEBUG,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    n_folds=Config.N_FOLDS,
    load_cached_data=False,
):
    """
    Orchestrates the K-Fold Cross-Validation training and submission generation.
    """
    seed_everything(Config.SEED)

    # --- 1. Load Data ---
    # df_train contains the merged development set (Fold 0)
    # df_test contains the test set (Fold 1)
    df_train, df_test = load_data()

    if debug:
        print("Debug mode: utilizing subset of data.")
        df_train = df_train.sample(n=50, random_state=Config.SEED).reset_index(
            drop=True
        )
        epochs = 2

    # Prepare for Stratification
    # We need X (dummy) and y (labels) for IterativeStratification
    X_dummy = np.zeros((len(df_train), 1))
    label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
    y_labels = df_train[label_cols].values

    # Initialize Splitter
    splitter = IterativeStratification(n_splits=n_folds, order=1)

    # Store fold scores
    fold_scores = []

    # --- 2. K-Fold Training Loop ---
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_dummy, y_labels)):
        print(f"\n{'='*20} Fold {fold+1}/{n_folds} {'='*20}")

        # Split DataFrames
        df_fold_train = df_train.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_train.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = BirdDataset(
            df_fold_train, transforms=get_transforms(data="train")
        )
        val_dataset = BirdDataset(df_fold_val, transforms=get_transforms(data="valid"))

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = BirdClassifier(
            backbone=Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
        )
        model.to(Config.DEVICE)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # Training Variables
        best_auc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")

        # Epoch Loop
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            # Train
            train_loss = train_one_epoch(
                model, optimizer, train_loader, Config.DEVICE, epoch
            )

            # Validate
            val_loss, _, val_targets = validate_one_epoch(
                model, val_loader, Config.DEVICE
            )

            # Step Scheduler
            scheduler.step()

            # Since validate_one_epoch prints metrics, we just need to capture the score
            # We need to re-calculate score or modify validate_one_epoch to return it.
            # Based on provided library.engine, validate_one_epoch returns (loss, preds, targets).
            # It prints the AUC but doesn't return it. We must recalculate it here for logic.
            # However, the library utils has calculate_roc_auc.

            # Re-calculate AUC for logic flow (since engine prints it but returns arrays)
            # To avoid re-inference, we use the returned predictions.
            # Wait, validate_one_epoch in library.engine returns (avg_loss, predictions, targets).
            # We can compute AUC from these.
            from library.utils import calculate_roc_auc

            # We need to ensure predictions are passed correctly.
            # validate_one_epoch returns probabilities (sigmoid applied).
            # calculate_roc_auc expects probabilities.

            # We need to rerun validation inference? No, validate_one_epoch returns predictions.
            # Let's call validate_one_epoch again? No, that's wasteful.
            # I will modify the logic to assume validate_one_epoch returns what is documented.
            # Documentation says: returns (average_loss, predictions, targets)

            _, val_preds, val_targets = validate_one_epoch(
                model, val_loader, Config.DEVICE
            )
            current_auc = calculate_roc_auc(val_targets, val_preds)

            print(f"Fold {fold+1} Epoch {epoch} - AUC: {current_auc}")

            # Early Stopping & Checkpointing
            if current_auc > best_auc:
                best_auc = current_auc
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_score": best_auc,
                    },
                    best_model_path,
                )
                print(f"New best model saved for Fold {fold+1} with AUC: {best_auc}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        fold_scores.append(best_auc)

        # Cleanup to save memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    print(f"\nTraining Complete. Average CV AUC: {np.mean(fold_scores)}")

    # --- 3. Inference & Submission ---
    create_submission(df_test, n_folds)


def create_submission(df_test, n_folds):
    """
    Generates predictions using the ensemble of trained fold models.
    """
    print("\nStarting Inference on Test Set...")

    test_dataset = BirdDataset(df_test, transforms=get_transforms(data="test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Array to store sum of predictions from all folds
    # Shape: (N_samples, N_classes)
    avg_preds = np.zeros((len(df_test), Config.NUM_CLASSES))

    for fold in range(n_folds):
        model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found. Skipping.")
            continue

        print(f"Predicting with model fold {fold}...")

        model = BirdClassifier(
            backbone=Config.BACKBONE,
            pretrained=False,  # Weights loaded from checkpoint
            num_classes=Config.NUM_CLASSES,
        )
        model.to(Config.DEVICE)

        # Load weights
        _, _ = load_checkpoint(model_path, model, device=Config.DEVICE)

        # Inference
        _, preds, _ = validate_one_epoch(model, test_loader, Config.DEVICE)

        avg_preds += preds

        del model
        torch.cuda.empty_cache()

    # Average predictions
    avg_preds /= n_folds

    # --- 4. Format Submission ---
    print("Formatting submission...")

    submission_rows = []

    # rec_ids from test dataframe
    rec_ids = df_test["rec_id"].values

    for i, rec_id in enumerate(rec_ids):
        probs = avg_preds[i]
        for species_idx, prob in enumerate(probs):
            # Construct Id as per competition format: rec_id * 100 + species_id
            row_id = int(rec_id * 100 + species_idx)
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_submission = pd.DataFrame(submission_rows)

    # Sort by Id just to be clean
    df_submission = df_submission.sort_values("Id")

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
