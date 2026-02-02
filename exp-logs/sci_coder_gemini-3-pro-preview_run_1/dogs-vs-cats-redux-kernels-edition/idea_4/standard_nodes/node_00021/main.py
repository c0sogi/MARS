import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
import cv2

# Import library modules
from library.config import CFG
from library.utils import (
    seed_everything,
    get_logger,
    AverageMeter,
    save_checkpoint,
    print_metrics,
)
from library.dataset import load_dataset_metadata, make_loader
from library.models import build_model
from library.engine import train_one_epoch, valid_one_epoch, predict_tta


def main():
    # 1. Setup
    seed_everything(CFG.seed)
    device = CFG.device

    # Limit data for fast baseline execution as per requirements
    # 8000 samples balances speed (approx 1 hour runtime) with performance
    MAX_SAMPLES = 8000

    print(f"Device: {device}")
    print(f"Model Architectures: {CFG.model_names}")

    # 2. Data Loading
    print("Loading metadata...")
    df_train = load_dataset_metadata(CFG.train_csv)
    df_test = load_dataset_metadata(CFG.test_csv)

    # Subsample training data
    if len(df_train) > MAX_SAMPLES:
        print(
            f"Subsampling training data from {len(df_train)} to {MAX_SAMPLES} samples."
        )
        df_train = df_train.sample(n=MAX_SAMPLES, random_state=CFG.seed).reset_index(
            drop=True
        )

    # 3. Prepare Folds
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    df_train["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train["label"])):
        df_train.loc[val_idx, "fold"] = fold

    # Initialize containers for ensemble predictions
    # OOF predictions: aligned with df_train
    oof_preds_accum = np.zeros(len(df_train))
    # Test predictions: aligned with df_test
    test_preds_accum = np.zeros(len(df_test))

    # 4. Training Loop
    # We iterate through each model architecture, then through each fold
    for model_name in CFG.model_names:
        print(f"\n=========================================")
        print(f"Processing Architecture: {model_name}")
        print(f"=========================================")

        for fold in range(CFG.n_folds):
            print(f"\n--- Fold {fold + 1}/{CFG.n_folds} ---")

            # Create DataLoaders for this fold
            train_df_fold = df_train[df_train["fold"] != fold].reset_index(drop=True)
            val_df_fold = df_train[df_train["fold"] == fold].reset_index(drop=True)

            train_loader = make_loader(
                train_df_fold,
                image_size=CFG.image_size,
                batch_size=CFG.batch_size,
                is_train=True,
            )
            valid_loader = make_loader(
                val_df_fold,
                image_size=CFG.image_size,
                batch_size=CFG.batch_size,
                is_train=False,
            )

            # Build Model
            model = build_model(model_name, pretrained=True)
            model.to(device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
            )

            # Train
            best_val_loss = float("inf")

            for epoch in range(CFG.epochs):
                print(f"[Epoch {epoch + 1}/{CFG.epochs}] Training...")
                train_loss = train_one_epoch(
                    model, optimizer, scheduler, train_loader, device, epoch
                )

                # Validate
                val_loss, val_log_loss, val_acc, _ = valid_one_epoch(
                    model, valid_loader, device
                )

                # Save Best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(
                        CFG.output_dir, f"{model_name}_fold_{fold}.pth"
                    )
                    save_checkpoint(model.state_dict(), True, save_path)

            # Load Best Model for Inference
            save_path = os.path.join(CFG.output_dir, f"{model_name}_fold_{fold}.pth")
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()

            # Generate OOF Predictions
            # Note: valid_one_epoch returns predictions in the order of the loader
            _, _, _, fold_oof_preds = valid_one_epoch(model, valid_loader, device)

            # Accumulate OOF predictions
            # Map predictions back to the original dataframe indices
            val_indices = df_train[df_train["fold"] == fold].index
            oof_preds_accum[val_indices] += fold_oof_preds

            # Generate Test Predictions (TTA)
            test_loader = make_loader(
                df_test,
                image_size=CFG.image_size,
                batch_size=CFG.batch_size,
                is_train=False,
            )
            fold_test_preds, _ = predict_tta(model, test_loader, device)
            test_preds_accum += fold_test_preds

            # Cleanup to save memory
            del model, optimizer, scheduler, train_loader, valid_loader, test_loader
            torch.cuda.empty_cache()

    # 5. Aggregation
    num_architectures = len(CFG.model_names)

    # Average OOF predictions: Sum / Num_Architectures
    # (Since each sample appears in exactly one fold per architecture)
    final_oof_preds = oof_preds_accum / num_architectures

    # Average Test predictions: Sum / (Num_Architectures * Num_Folds)
    final_test_preds = test_preds_accum / (num_architectures * CFG.n_folds)

    # Final Metric
    final_log_loss = log_loss(df_train["label"], final_oof_preds)
    print(f"Final Validation Metric: {final_log_loss}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    df_train["pred"] = final_oof_preds
    df_train["error"] = (df_train["label"] - df_train["pred"]).abs()

    # Extract metadata features for correlation analysis
    print("Extracting image features for analysis...")
    file_sizes = []
    widths = []
    heights = []

    for rel_path in df_train["filepath"]:
        full_path = os.path.join(CFG.input_dir, rel_path)
        try:
            # File size
            file_sizes.append(os.path.getsize(full_path))

            # Dimensions
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        except Exception:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    df_train["file_size"] = file_sizes
    df_train["width"] = widths
    df_train["height"] = heights

    # Calculate Correlations
    features = ["file_size", "width", "height"]
    for feat in features:
        if df_train[feat].std() > 0:
            corr, _ = pearsonr(df_train["error"], df_train[feat])
            print(f"Correlation between error and {feat}: {corr}")
        else:
            print(f"Correlation between error and {feat}: NaN (constant feature)")

    # 7. Submission
    THRESHOLD = 0.008227705646706841

    if final_log_loss < THRESHOLD:
        print(f"\nMetric {final_log_loss} is better than threshold {THRESHOLD}.")
        print("Generating submission file...")

        submission = pd.DataFrame({"id": df_test["id"], "label": final_test_preds})

        submission_path = os.path.join(CFG.submission_dir, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(f"\nMetric {final_log_loss} did not meet threshold {THRESHOLD}.")
        print("Submission skipped.")


if __name__ == "__main__":
    main()
