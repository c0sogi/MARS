import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from library.config import CFG
from library.utils import (
    seed_everything,
    calculate_class_weights,
    save_submission,
    calculate_roc_auc,
)
from library.dataset import AppleDataset, get_transforms, prepare_folds
from library.model import AppleResNet34, verify_initialization
from library.engine import train_one_epoch, valid_one_epoch


def run_calibration_phase(load_cached_data=True):
    """
    Executes Phase 1: Proxy Calibration using 5-Fold CV.
    Determines the Global Optimal Epoch (E_opt) where validation AUC peaks.

    Args:
        load_cached_data (bool): Whether to load prepared folds/weights from cache.

    Returns:
        int: The optimal number of epochs (E_opt).
    """
    print("\n" + "=" * 40)
    print("PHASE 1: PROXY CALIBRATION (5-FOLD CV)")
    print("=" * 40)

    seed_everything(CFG.seed)

    # Load Data (Stratified Folds)
    df = prepare_folds(load_cached_data=load_cached_data)

    # Calculate Class Weights for Loss Balancing
    class_weights = calculate_class_weights(
        df, CFG.target_cols, load_cached_data=load_cached_data
    )

    # Matrix to store AUC per fold per epoch
    # Shape: (n_folds, n_epochs)
    fold_aucs = np.zeros((CFG.n_folds, CFG.calibration_epochs))

    for fold in range(CFG.n_folds):
        print(f"\n--- Fold {fold + 1}/{CFG.n_folds} ---")

        # Split Data
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)

        # Setup Datasets
        train_dataset = AppleDataset(train_df, transform=get_transforms("train"))
        valid_dataset = AppleDataset(valid_df, transform=get_transforms("valid"))

        # Setup Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        model = AppleResNet34(pretrained=True)
        model.to(CFG.device)

        # Loss, Optimizer, Scheduler
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

        # Scheduler synchronized to calibration budget
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=CFG.calibration_epochs, T_mult=1, eta_min=CFG.min_lr
        )

        # Initialization Verification (Safeguard)
        verify_initialization(model, train_loader, criterion, CFG.device)

        # Training Loop
        for epoch in range(CFG.calibration_epochs):
            print(f"Fold {fold+1} | Epoch {epoch+1}/{CFG.calibration_epochs}")

            _ = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, CFG.device
            )
            _, valid_preds, valid_labels = valid_one_epoch(
                model, valid_loader, criterion, CFG.device
            )

            # Calculate and store AUC
            epoch_auc = calculate_roc_auc(valid_labels, valid_preds)
            fold_aucs[fold, epoch] = epoch_auc

        # Cleanup to free memory
        del model, optimizer, scheduler, train_loader, valid_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Aggregate Results
    mean_aucs = fold_aucs.mean(axis=0)
    best_epoch_idx = np.argmax(mean_aucs)
    best_epoch = int(best_epoch_idx + 1)
    best_auc = mean_aucs[best_epoch_idx]

    print("\nCalibration Results (Mean AUC per Epoch):")
    for i, auc in enumerate(mean_aucs):
        print(f"Epoch {i+1}: {auc:.6f}")

    print(f"\nGlobal Optimal Epoch (E_opt): {best_epoch} (AUC: {best_auc:.6f})")

    return best_epoch


def run_production_phase(optimal_epoch, load_cached_data=True):
    """
    Executes Phase 2: Production Training (Seed Ensemble).
    Trains 5 models on 100% of data for exactly E_opt epochs.

    Args:
        optimal_epoch (int): The number of epochs to train (derived from Phase 1).
        load_cached_data (bool): Whether to load prepared data from cache.
    """
    print("\n" + "=" * 40)
    print(f"PHASE 2: PRODUCTION TRAINING (E_opt={optimal_epoch})")
    print("=" * 40)

    # Load Full Data (Combine Train + Val)
    df = prepare_folds(load_cached_data=load_cached_data)

    # Calculate Class Weights
    class_weights = calculate_class_weights(
        df, CFG.target_cols, load_cached_data=load_cached_data
    )

    # Train one model per seed
    for i, seed in enumerate(CFG.ensemble_seeds):
        print(
            f"\n--- Training Seed Model {i+1}/{len(CFG.ensemble_seeds)} (Seed: {seed}) ---"
        )

        seed_everything(seed)

        # Full Dataset (100% Data)
        train_dataset = AppleDataset(df, transform=get_transforms("train"))
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Initialize Model
        model = AppleResNet34(pretrained=True)
        model.to(CFG.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

        # Scheduler synchronized to optimal_epoch
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=optimal_epoch, T_mult=1, eta_min=CFG.min_lr
        )

        # Initialization Verification
        verify_initialization(model, train_loader, criterion, CFG.device)

        # Training Loop (No Validation)
        for epoch in range(optimal_epoch):
            print(f"Seed {seed} | Epoch {epoch+1}/{optimal_epoch}")
            _ = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, CFG.device
            )

        # Save Model Checkpoint
        model_name = f"resnet34_seed_{seed}.pth"
        save_path = os.path.join(CFG.models_dir, model_name)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Cleanup
        del model, optimizer, scheduler, train_loader
        torch.cuda.empty_cache()
        gc.collect()


def generate_submission():
    """
    Generates predictions for the test set using the ensemble of models.
    Saves the result to submission.csv.
    """
    print("\n" + "=" * 40)
    print("INFERENCE & SUBMISSION")
    print("=" * 40)

    # Load Test Data
    test_df = pd.read_csv(CFG.test_metadata_path)
    test_dataset = AppleDataset(test_df, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Identify models
    model_files = [f for f in os.listdir(CFG.models_dir) if f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError(f"No models found in {CFG.models_dir}")

    ensemble_preds = []

    # Inference Loop
    for model_file in model_files:
        print(f"Predicting with {model_file}...")

        model_path = os.path.join(CFG.models_dir, model_file)

        # Load Model
        model = AppleResNet34(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=CFG.device))
        model.to(CFG.device)
        model.eval()

        preds = []
        with torch.no_grad():
            for images, _, _ in test_loader:
                images = images.to(CFG.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        ensemble_preds.append(preds)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Average Predictions (Ensemble Aggregation)
    avg_preds = np.mean(ensemble_preds, axis=0)

    # Save Submission
    save_submission(
        avg_preds, test_df["image_id"].values, CFG.target_cols, CFG.submission_path
    )
    print(f"Submission saved to {CFG.submission_path}")
