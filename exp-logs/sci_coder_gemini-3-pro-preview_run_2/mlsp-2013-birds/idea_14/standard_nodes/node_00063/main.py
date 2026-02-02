import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from skmultilearn.model_selection import IterativeStratification

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_pos_weights
from library.dataset import get_dataloaders
from library.models import BirdModel
from library.engine import train_one_epoch, validate, inference, EarlyStopping


def run_fold_training(model_name, train_df, val_df, fold_idx):
    """
    Trains a model for a specific fold.
    """
    # Setup DataLoaders
    # Disable caching to prevent conflicts between folds sharing the same cache filename
    dataloaders = get_dataloaders(
        model_name=model_name,
        train_df=train_df,
        val_df=val_df,
        load_cached_data=False,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Initialize Model
    device = Config.DEVICE
    model = BirdModel(model_name, pretrained=True).to(device)

    # Loss Function (Weighted BCE)
    pos_weights = get_pos_weights(train_df, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Early Stopping
    early_stopping = EarlyStopping(patience=10, mode="max")

    best_auc = 0.0
    model_save_path = os.path.join(
        Config.WORKING_DIR, f"model_{model_name}_fold_{fold_idx}.pth"
    )

    # Training Loop
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_save_path)

        early_stopping(val_auc)
        if early_stopping.early_stop:
            break

    return best_auc, model_save_path


def main():
    # 1. Setup
    Config.setup()
    device = Config.DEVICE

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Combine for CV
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    # Prepare for Stratified Split
    label_cols = [c for c in dev_df.columns if c.startswith("species_")]
    X = dev_df["rec_id"].values.reshape(-1, 1)
    y = dev_df[label_cols].values

    # Models to use
    models = Config.MODELS_TO_RUN

    print(f"Starting execution on {device}...")
    print("\n--- Starting 5-Fold Cross-Validation ---")

    k_fold = IterativeStratification(n_splits=5, order=1)

    # Store OOF predictions
    oof_preds = np.zeros((len(dev_df), Config.NUM_CLASSES))
    oof_targets = y

    # Store model paths
    model_paths = {m: [] for m in models}

    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        print(f"\nFold {fold_idx + 1}/5")

        fold_train_df = dev_df.iloc[train_indices].reset_index(drop=True)
        fold_val_df = dev_df.iloc[val_indices].reset_index(drop=True)

        fold_preds = []  # To average across models for this fold

        for model_name in models:
            best_auc, save_path = run_fold_training(
                model_name, fold_train_df, fold_val_df, fold_idx
            )
            model_paths[model_name].append(save_path)

            # Load best model for inference
            model = BirdModel(model_name, pretrained=False).to(device)
            model.load_state_dict(torch.load(save_path, map_location=device))

            # Get Val Loader (uncached to avoid conflicts)
            dataloaders = get_dataloaders(
                model_name, val_df=fold_val_df, load_cached_data=False
            )
            val_loader = dataloaders["val"]

            preds = inference(model, val_loader, device)
            fold_preds.append(preds)

        # Average predictions from all models for this fold
        avg_fold_preds = np.mean(fold_preds, axis=0)
        oof_preds[val_indices] = avg_fold_preds

    # -------------------------------------------------------------------------
    # FINAL EVALUATION
    # -------------------------------------------------------------------------
    print("\n--- Final Evaluation (OOF) ---")

    from sklearn.metrics import roc_auc_score

    aucs = []
    for i in range(Config.NUM_CLASSES):
        if len(np.unique(oof_targets[:, i])) > 1:
            aucs.append(roc_auc_score(oof_targets[:, i], oof_preds[:, i]))

    final_metric = np.mean(aucs) if aucs else 0.5
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # FAILURE ANALYSIS
    # -------------------------------------------------------------------------
    sample_errors = np.mean(np.abs(oof_targets - oof_preds), axis=1)
    cardinality = np.sum(oof_targets, axis=1)

    corr_cardinality, _ = pearsonr(sample_errors, cardinality)
    print(f"Correlation between Error and Label Cardinality: {corr_cardinality:.4f}")

    # -------------------------------------------------------------------------
    # SUBMISSION GENERATION
    # -------------------------------------------------------------------------
    threshold = 0.9129501920716607

    if final_metric > threshold:
        print("Metric threshold met. Generating submission...")
        # Generate Test Predictions with Ensemble of ALL models (5 folds * N architectures)
        test_preds_all = []

        for model_name in models:
            # We have 5 models per architecture
            for fold_idx, path in enumerate(model_paths[model_name]):
                model = BirdModel(model_name, pretrained=False).to(device)
                model.load_state_dict(torch.load(path, map_location=device))

                dataloaders = get_dataloaders(
                    model_name,
                    test_df=test_df,
                    load_cached_data=True,  # Cache is fine for test set
                )
                test_loader = dataloaders["test"]

                preds = inference(model, test_loader, device)
                test_preds_all.append(preds)

        ensemble_test_preds = np.mean(test_preds_all, axis=0)

        # Format Submission
        submission_rows = []
        rec_ids = test_df["rec_id"].values

        for idx, rec_id in enumerate(rec_ids):
            probs = ensemble_test_preds[idx]
            for species_id, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_id)
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_df = submission_df.sort_values("Id")

        os.makedirs("submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
