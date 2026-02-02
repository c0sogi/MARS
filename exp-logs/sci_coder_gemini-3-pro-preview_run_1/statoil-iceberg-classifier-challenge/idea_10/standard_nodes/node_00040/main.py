import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Import from library
from library.config import Config
from library.utils import seed_everything, log_metric, calculate_log_loss
from library.data import get_dataloaders, get_test_loader, load_data
from library.model import IcebergResNet18
from library.engine import train_one_epoch, validate, predict_tta


def run_calibration():
    """
    Executes Phase 1: Stratified 5-Fold Cross-Validation.
    Determines the optimal number of training epochs and the expected validation performance.
    """
    print("Starting Phase 1: Calibration (5-Fold CV)...")

    # Matrix to store validation metrics: [Epochs, Folds]
    val_losses = np.zeros((Config.CALIBRATION_EPOCHS, Config.N_FOLDS))

    # Loop through folds
    for fold in range(Config.N_FOLDS):
        print(f"  Running Fold {fold}/{Config.N_FOLDS - 1}")
        seed_everything(Config.SEED)

        # Get DataLoaders (Cached)
        train_loader, val_loader = get_dataloaders(
            fold_index=fold, full_fit=False, load_cached_data=True
        )

        # Initialize Model
        model = IcebergResNet18().to(Config.DEVICE)

        # Initialize Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler (Reactive ReduceLROnPlateau)
        # Cite solution_lesson_node_00038: Use Reactive Scheduling
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )

        # Training Loop
        for epoch in range(Config.CALIBRATION_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, Config.DEVICE, epoch
            )
            epoch_loss, val_metric = validate(model, val_loader, Config.DEVICE)

            # Step Scheduler with Validation Metric
            scheduler.step(val_metric)

            # Store metric (Log Loss)
            val_losses[epoch, fold] = val_metric

    # --- Analysis of CV Results ---
    # Calculate average validation loss per epoch across all folds
    avg_val_losses = np.mean(val_losses, axis=1)

    # Find the epoch with the minimum average validation loss
    optimal_epoch_idx = np.argmin(avg_val_losses)
    optimal_epochs = optimal_epoch_idx + 1
    best_cv_score = avg_val_losses[optimal_epoch_idx]

    print(f"  Optimal Epochs Found: {optimal_epochs}")
    print(f"  Best CV Score (Avg Log Loss): {best_cv_score:.6f}")

    # --- Preparation for Failure Analysis ---
    # To perform failure analysis, we need predictions and labels.
    # We retrain Fold 0 for exactly optimal_epochs to get a representative set of validation predictions.
    print("  Retraining Fold 0 for Failure Analysis...")
    seed_everything(Config.SEED)
    train_loader, val_loader = get_dataloaders(
        fold_index=0, full_fit=False, load_cached_data=True
    )

    model = IcebergResNet18().to(Config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=optimal_epochs, eta_min=0)

    for epoch in range(optimal_epochs):
        train_one_epoch(model, train_loader, optimizer, Config.DEVICE, epoch)
        scheduler.step()

    # Generate predictions on validation set
    model.eval()
    preds = []
    targets = []
    angles = []

    with torch.no_grad():
        for imgs, angs, lbls in val_loader:
            imgs = imgs.to(Config.DEVICE)
            angs = angs.to(Config.DEVICE)

            logits = model(imgs, angs)
            probs = torch.sigmoid(logits).cpu().numpy()

            preds.extend(probs)
            targets.extend(lbls.numpy())
            angles.extend(angs.cpu().numpy())

    oof_data = {
        "pred": np.concatenate(preds).flatten(),
        "label": np.array(targets).flatten(),
        "angle": np.array(angles).flatten(),
    }

    return optimal_epochs, best_cv_score, oof_data


def analyze_failures(oof_data):
    """
    Performs failure analysis on the validation set predictions.
    Calculates correlation between error and incidence angle.
    """
    print("\nFailure Analysis (Representative Fold):")
    df = pd.DataFrame(oof_data)

    # Calculate absolute error
    df["error"] = np.abs(df["pred"] - df["label"])

    # Calculate correlation
    # Note: 'angle' is normalized, but correlation is invariant to linear scaling
    corr = df["error"].corr(df["angle"])
    print(f"Correlation between Error and Incidence Angle: {corr:.10f}")

    # Stats
    mean_error = df["error"].mean()
    print(f"Mean Absolute Error: {mean_error:.4f}")


def run_production(optimal_epochs):
    """
    Phase 2: Full-Fit Seed Ensemble.
    Trains 5 independent models on the entire dataset (Train + Val) for E_opt epochs.
    """
    print("Starting Phase 2: Production (Full-Fit Ensemble)...")
    models_list = []

    for i, seed in enumerate(Config.ENSEMBLE_SEEDS):
        print(f"  Training Model {i+1}/{len(Config.ENSEMBLE_SEEDS)} (Seed {seed})")
        seed_everything(seed)

        # Get Full-Fit DataLoader (100% Data)
        train_loader, _ = get_dataloaders(full_fit=True, load_cached_data=True)

        # Initialize Model
        model = IcebergResNet18().to(Config.DEVICE)

        # Initialize Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler (Cosine Decay over optimal_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=optimal_epochs, eta_min=0)

        # Training Loop
        for epoch in range(optimal_epochs):
            train_one_epoch(model, train_loader, optimizer, Config.DEVICE, epoch)
            scheduler.step()

        models_list.append(model)

    return models_list


def main():
    # Ensure cache directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Run Calibration (Phase 1)
    optimal_epochs, cv_score, oof_data = run_calibration()

    # 2. Report Validation Metric (Required)
    # We use the best CV score as the robust estimate of validation performance
    print(f"Final Validation Metric: {cv_score}")

    # 3. Perform Failure Analysis (Required)
    analyze_failures(oof_data)

    # 4. Check Threshold and Generate Submission
    # Threshold from prompt: 0.17822679498532543
    THRESHOLD = 0.17822679498532543

    if cv_score < THRESHOLD:
        print(
            f"Validation metric {cv_score:.6f} is lower than {THRESHOLD}. Proceeding to submission."
        )

        # 5. Run Production Training (Phase 2)
        models = run_production(optimal_epochs)

        # 6. Inference with TTA
        print("Generating predictions on Test Set...")
        test_loader, test_ids = get_test_loader(load_cached_data=True)

        # Accumulate predictions from all models
        final_preds = np.zeros(len(test_ids))

        for i, model in enumerate(models):
            # predict_tta handles the 3-view TTA (Original, HFlip, VFlip) internally
            preds = predict_tta(model, test_loader, Config.DEVICE)
            final_preds += preds

        # Average across the 5 models
        final_preds /= len(models)

        # 7. Save Submission
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": final_preds})

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {cv_score:.6f} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
