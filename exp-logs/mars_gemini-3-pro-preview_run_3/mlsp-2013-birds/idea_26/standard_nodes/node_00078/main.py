import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_robust_auc, CheckpointManager
from library.data import (
    prepare_folds,
    get_dataloaders,
    get_test_dataloaders,
    load_histogram_features,
)
from library.models import BirdCNN, BirdMLP
from library.engine import train_one_epoch, validate


def run():
    # 1. Setup
    # Cite debug_lesson_13: Prevent Artifact Pollution by Enforcing Strict File Search Patterns
    if os.path.exists(Config.CHECKPOINT_DIR):
        print(f"Cleaning stale checkpoints from {Config.CHECKPOINT_DIR}")
        shutil.rmtree(Config.CHECKPOINT_DIR)

    Config.setup()
    seed_everything(Config.SEED)

    print(f"Device: {Config.DEVICE}")

    # 2. Data Preparation
    # Load and stratify data into folds
    df_folds = prepare_folds(load_cached_data=True)

    # Prepare storage for Out-Of-Fold (OOF) predictions for global validation
    # We map rec_id to a row index to fill predictions correctly
    rec_id_to_idx = {row["rec_id"]: idx for idx, row in df_folds.iterrows()}
    num_samples = len(df_folds)
    oof_preds = np.zeros((num_samples, Config.NUM_CLASSES))
    oof_targets = np.zeros((num_samples, Config.NUM_CLASSES))

    # 3. Training Loop
    for fold in range(Config.NUM_FOLDS):
        print(f"\n{'='*20} Fold {fold}/{Config.NUM_FOLDS - 1} {'='*20}")

        # Get dataloaders for this fold
        loaders = get_dataloaders(fold, df_folds)

        # --- A. Train Deep Learning Stream (CNNs) ---
        for arch in Config.CNN_ARCHITECTURES:
            print(f"Training CNN Architecture: {arch}")

            # Initialize Model, Optimizer, Scheduler
            model = BirdCNN(arch, pretrained=True).to(Config.DEVICE)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LR_CNN,
                weight_decay=Config.WEIGHT_DECAY_CNN,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=Config.MAX_EPOCHS)

            # Checkpoint Manager to keep Top-K models
            ckpt_manager = CheckpointManager(arch, fold)

            patience_counter = 0
            best_fold_auc = 0.0

            for epoch in range(1, Config.MAX_EPOCHS + 1):
                # Train
                train_loss = train_one_epoch(
                    model,
                    loaders["cnn"]["train"],
                    optimizer,
                    scheduler,
                    Config.DEVICE,
                    epoch,
                )

                # Validate
                val_loss, val_auc, _, _ = validate(
                    model, loaders["cnn"]["val"], Config.DEVICE
                )

                # Save Checkpoint
                saved = ckpt_manager.save(model, val_auc, epoch)

                if saved:
                    patience_counter = 0
                    best_fold_auc = max(best_fold_auc, val_auc)
                else:
                    patience_counter += 1

                # Early Stopping
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping {arch} at epoch {epoch} (Best AUC: {best_fold_auc:.4f})"
                    )
                    break

        # --- B. Train Shallow Learning Stream (MLP) ---
        print("Training MLP Architecture: Bag-of-Words")
        model = BirdMLP().to(Config.DEVICE)
        optimizer = optim.Adam(
            model.parameters(), lr=Config.LR_MLP, weight_decay=Config.WEIGHT_DECAY_MLP
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.MAX_EPOCHS)

        ckpt_manager = CheckpointManager("mlp", fold)

        patience_counter = 0
        best_fold_auc = 0.0

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            train_loss = train_one_epoch(
                model,
                loaders["mlp"]["train"],
                optimizer,
                scheduler,
                Config.DEVICE,
                epoch,
            )
            val_loss, val_auc, _, _ = validate(
                model, loaders["mlp"]["val"], Config.DEVICE
            )

            saved = ckpt_manager.save(model, val_auc, epoch)

            if saved:
                patience_counter = 0
                best_fold_auc = max(best_fold_auc, val_auc)
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping MLP at epoch {epoch} (Best AUC: {best_fold_auc:.4f})"
                )
                break

        # --- C. Generate OOF Predictions for this Fold ---
        print(f"Generating OOF predictions for Fold {fold}...")

        # Identify validation indices
        val_rec_ids = df_folds[df_folds["fold"] == fold]["rec_id"].values
        val_indices = [rec_id_to_idx[rid] for rid in val_rec_ids]

        fold_ensemble_preds = []

        # Iterate over all architectures (CNNs + MLP)
        all_archs = Config.CNN_ARCHITECTURES + ["mlp"]

        for arch in all_archs:
            # Find all saved checkpoints for this model/fold
            pattern = os.path.join(Config.CHECKPOINT_DIR, f"{arch}_fold_{fold}_*.pth")
            checkpoints = glob.glob(pattern)

            # Load appropriate validation loader
            if arch == "mlp":
                loader = loaders["mlp"]["val"]
            else:
                loader = loaders["cnn"]["val"]

            for ckpt_path in checkpoints:
                # Re-instantiate model structure
                if arch == "mlp":
                    model = BirdMLP().to(Config.DEVICE)
                else:
                    model = BirdCNN(arch, pretrained=False).to(Config.DEVICE)

                # Load weights
                model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))

                # Predict
                _, _, preds, targets = validate(model, loader, Config.DEVICE)
                fold_ensemble_preds.append(preds)

                # Store targets (overwrite is fine as they are identical for the fold)
                oof_targets[val_indices] = targets

        # Average predictions across all models/snapshots for this fold
        if fold_ensemble_preds:
            avg_fold_preds = np.mean(fold_ensemble_preds, axis=0)
            oof_preds[val_indices] = avg_fold_preds

    # 4. Global Evaluation & Failure Analysis
    print("\n" + "=" * 40)
    print("Evaluation & Failure Analysis")
    print("=" * 40)

    # Calculate global metric
    final_auc = calculate_robust_auc(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # 1. Calculate Error Magnitude (BCE per sample, averaged over classes)
    epsilon = 1e-7
    preds_clipped = np.clip(oof_preds, epsilon, 1 - epsilon)
    # BCE = - [y * log(p) + (1-y) * log(1-p)]
    bce_matrix = -(
        oof_targets * np.log(preds_clipped)
        + (1 - oof_targets) * np.log(1 - preds_clipped)
    )
    error_magnitude = np.mean(bce_matrix, axis=1)  # Shape: (num_samples,)

    # 2. Correlate with Label Count (Complexity)
    label_counts = np.sum(oof_targets, axis=1)
    corr_labels = np.corrcoef(error_magnitude, label_counts)[0, 1]

    # 3. Correlate with Feature Norm (Signal Energy Proxy)
    feature_map = load_histogram_features()
    feature_norms = []
    for _, row in df_folds.iterrows():
        rid = row["rec_id"]
        feat = feature_map.get(rid, np.zeros(Config.MLP_INPUT_DIM))
        feature_norms.append(np.linalg.norm(feat))

    corr_feat_norm = np.corrcoef(error_magnitude, feature_norms)[0, 1]

    print("Failure Analysis Correlations (Error Magnitude vs Feature):")
    print(f"  vs Label Count: {corr_labels:.6f}")
    print(f"  vs Feature Norm: {corr_feat_norm:.6f}")

    # 5. Inference & Submission
    threshold = 0.9479806884980326
    if final_auc > threshold:
        print(f"\nThreshold ({threshold}) met. Generating submission...")

        # Get Test Loaders
        test_data = get_test_dataloaders()
        test_ids = test_data["ids"]

        test_preds_accumulator = []

        # Iterate over all folds and all models
        for fold in range(Config.NUM_FOLDS):
            for arch in Config.CNN_ARCHITECTURES + ["mlp"]:
                pattern = os.path.join(
                    Config.CHECKPOINT_DIR, f"{arch}_fold_{fold}_*.pth"
                )
                checkpoints = glob.glob(pattern)

                if arch == "mlp":
                    loader = test_data["mlp"]
                else:
                    loader = test_data["cnn"]

                for ckpt_path in checkpoints:
                    # Load model
                    if arch == "mlp":
                        model = BirdMLP().to(Config.DEVICE)
                    else:
                        model = BirdCNN(arch, pretrained=False).to(Config.DEVICE)

                    model.load_state_dict(
                        torch.load(ckpt_path, map_location=Config.DEVICE)
                    )
                    model.eval()

                    # Predict
                    batch_preds = []
                    with torch.no_grad():
                        for inputs, _ in loader:
                            inputs = inputs.to(Config.DEVICE)
                            outputs = model(inputs)
                            probs = torch.sigmoid(outputs)
                            batch_preds.append(probs.cpu().numpy())

                    if batch_preds:
                        test_preds_accumulator.append(
                            np.concatenate(batch_preds, axis=0)
                        )

        # Average all predictions
        if test_preds_accumulator:
            final_test_preds = np.mean(test_preds_accumulator, axis=0)

            # Format Submission
            submission_rows = []
            for i, rec_id in enumerate(test_ids):
                probs = final_test_preds[i]
                for species_id in range(Config.NUM_CLASSES):
                    # Id format: rec_id * 100 + species_number
                    row_id = int(rec_id * 100 + species_id)
                    prob = probs[species_id]
                    submission_rows.append({"Id": row_id, "Probability": prob})

            sub_df = pd.DataFrame(submission_rows)
            sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {final_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
