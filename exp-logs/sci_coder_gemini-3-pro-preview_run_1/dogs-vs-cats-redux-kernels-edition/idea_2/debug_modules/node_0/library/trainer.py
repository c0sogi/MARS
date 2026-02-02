import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, save_checkpoint, save_submission
from library.dataset import DogCatDataset, get_transforms
from library.model import DogCatClassifier, train_one_epoch, validate, predict_test


def run_training_pipeline(
    debug: bool = Config.debug,
    epochs: int = Config.epochs,
    n_folds: int = Config.n_folds,
    batch_size: int = Config.batch_size,
    learning_rate: float = Config.learning_rate,
    weight_decay: float = Config.weight_decay,
):
    """
    Executes the 5-Fold Cross-Validation training pipeline for Dog vs Cat classification.

    Args:
        debug (bool): If True, uses a small subset of data for quick testing.
        epochs (int): Number of training epochs per fold.
        n_folds (int): Number of Cross-Validation folds.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Initial learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
    """
    seed_everything(Config.seed)
    device = torch.device(Config.device)
    print(f"Starting training pipeline on device: {device}")

    # --- Data Preparation ---
    # Load metadata for training and validation
    df_train_part = pd.read_csv(Config.train_metadata_path)
    df_val_part = pd.read_csv(Config.val_metadata_path)

    # Combine datasets to use 100% of labeled data for Cross-Validation
    df_train_full = pd.concat([df_train_part, df_val_part]).reset_index(drop=True)

    # Load test metadata
    df_test = pd.read_csv(Config.test_metadata_path)

    # Handle Debug Mode
    if debug:
        print("DEBUG MODE ENABLED: Subsampling data.")
        df_train_full = df_train_full.sample(
            n=200, random_state=Config.seed
        ).reset_index(drop=True)
        df_test = df_test.sample(n=50, random_state=Config.seed).reset_index(drop=True)

    # Prepare Test Loader (Fixed for all folds)
    test_dataset = DogCatDataset(
        df_test, transforms=get_transforms("test"), mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # --- 5-Fold Stratified Cross-Validation ---
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.seed)

    # Array to accumulate test predictions from each fold (Ensembling)
    fold_test_preds = np.zeros(len(df_test))
    test_ids = None

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["label"])
    ):
        print(f"\n{'='*20} Fold {fold + 1}/{n_folds} {'='*20}")

        # Create Fold Splits
        df_train_fold = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df_train_full.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_ds = DogCatDataset(
            df_train_fold, transforms=get_transforms("train"), mode="train"
        )
        val_ds = DogCatDataset(
            df_val_fold, transforms=get_transforms("valid"), mode="val"
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        model = DogCatClassifier().to(device)

        # Optimizer, Scheduler, Loss, Scaler
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=Config.min_lr
        )
        scaler = GradScaler()

        # Training Loop
        best_val_loss = float("inf")
        best_model_filename = f"model_fold_{fold+1}.pth"

        for epoch in range(epochs):
            # Train for one epoch
            train_loss = train_one_epoch(
                train_loader, model, criterion, optimizer, device, scaler, epoch
            )

            # Validate
            val_loss = validate(val_loader, model, criterion, device)

            # Step Scheduler
            scheduler.step()

            print(
                f"Fold {fold+1} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Save Best Model (Early Stopping Logic)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    {"state_dict": model.state_dict(), "val_loss": val_loss},
                    is_best=True,
                    filename=best_model_filename,
                )

        print(f"Fold {fold+1} Finished. Best Val Loss: {best_val_loss:.6f}")

        # --- Inference for this Fold ---
        # Load best weights
        best_model_path = os.path.join(Config.model_dir, best_model_filename)
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        # Predict on Test Set
        ids, preds = predict_test(test_loader, model, device)

        # Accumulate predictions
        fold_test_preds += preds
        if test_ids is None:
            test_ids = ids

        # Cleanup to free memory
        del model, optimizer, scaler, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # --- Ensemble Aggregation ---
    # Average predictions across all folds
    avg_preds = fold_test_preds / n_folds

    # --- Save Submission ---
    output_path = os.path.join(Config.submission_dir, "submission.csv")
    save_submission(test_ids, avg_preds, output_path=output_path)
    print(f"\nEnsemble prediction complete. Submission saved to {output_path}")
