import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import provided library functions
from library.utils import set_seed, calculate_pos_weights
from library.model import MILResNet18
from library.data_loader import BirdDataset
from library.train_eval import train_fn, eval_fn, inference_fn


def main():
    # --- Configuration ---
    CONFIG = {
        "seed": 42,
        "input_dir": "./input",
        "metadata_dir": "./metadata",
        "working_dir": "./working/run_baseline",
        "submission_dir": "./submission",
        "num_folds": 5,
        "epochs": 15,  # Sufficient for convergence on small dataset
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "mixup_alpha": 0.4,
        "image_size": 224,
        "patience": 5,
        "threshold": 0.8739452549958209,
    }

    # Ensure directories exist
    os.makedirs(CONFIG["working_dir"], exist_ok=True)
    os.makedirs(CONFIG["submission_dir"], exist_ok=True)

    # Set reproducibility
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    # Load metadata
    df_train_dev = pd.read_csv(os.path.join(CONFIG["metadata_dir"], "train.csv"))
    df_val_holdout = pd.read_csv(os.path.join(CONFIG["metadata_dir"], "val.csv"))
    df_test = pd.read_csv(os.path.join(CONFIG["metadata_dir"], "test.csv"))

    # Identify label columns
    label_cols = [c for c in df_train_dev.columns if c.startswith("species_")]
    num_classes = len(label_cols)

    # Prepare stratification targets (string representation of label sets)
    y_labels = df_train_dev[label_cols].values
    y_str = ["".join(map(str, row.astype(int))) for row in y_labels]

    # --- 2. Training (Stratified K-Fold on Dev Set) ---
    skf = StratifiedKFold(
        n_splits=CONFIG["num_folds"], shuffle=True, random_state=CONFIG["seed"]
    )

    model_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train_dev, y_str)):
        print(f"\n=== Training Fold {fold} ===")

        # Split dev set into fold-train and fold-val
        df_fold_train = df_train_dev.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_train_dev.iloc[val_idx].reset_index(drop=True)

        # Datasets & Loaders
        train_ds = BirdDataset(
            df_fold_train,
            CONFIG["input_dir"],
            image_size=CONFIG["image_size"],
            train=True,
        )
        val_ds = BirdDataset(
            df_fold_val,
            CONFIG["input_dir"],
            image_size=CONFIG["image_size"],
            train=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Model Setup
        model = MILResNet18(num_classes=num_classes, pretrained=True).to(device)

        # Loss (Weighted BCE)
        y_train_tensor = torch.tensor(df_fold_train[label_cols].values)
        pos_weights = calculate_pos_weights(y_train_tensor, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        optimizer = optim.AdamW(
            model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"]
        )

        # Training Loop
        best_auc = -1.0
        best_model_path = os.path.join(CONFIG["working_dir"], f"model_fold_{fold}.pth")
        model_paths.append(best_model_path)
        early_stop_counter = 0

        for epoch in range(CONFIG["epochs"]):
            train_loss = train_fn(
                model, train_loader, criterion, optimizer, device, CONFIG["mixup_alpha"]
            )
            val_loss, val_auc = eval_fn(model, val_loader, criterion, device)
            scheduler.step()

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= CONFIG["patience"]:
                break

        print(f"Fold {fold} Best AUC: {best_auc:.5f}")

    # --- 3. Hold-Out Validation & Ensemble Inference ---
    print("\n=== Performing Hold-Out Validation ===")

    # Prepare Hold-Out Loader
    holdout_ds = BirdDataset(
        df_val_holdout,
        CONFIG["input_dir"],
        image_size=CONFIG["image_size"],
        train=False,
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Ensemble Prediction
    holdout_preds_sum = np.zeros((len(df_val_holdout), num_classes))

    for path in model_paths:
        model = MILResNet18(num_classes=num_classes, pretrained=False)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)

        preds = inference_fn(model, holdout_loader, device)
        holdout_preds_sum += preds

    avg_holdout_preds = holdout_preds_sum / CONFIG["num_folds"]
    y_holdout_true = df_val_holdout[label_cols].values

    # Calculate Metric
    auc_scores = []
    for i in range(num_classes):
        if len(np.unique(y_holdout_true[:, i])) > 1:
            auc_scores.append(
                roc_auc_score(y_holdout_true[:, i], avg_holdout_preds[:, i])
            )

    if len(auc_scores) > 0:
        final_metric = np.mean(auc_scores)
    else:
        final_metric = 0.5

    print(f"Final Validation Metric: {final_metric}")

    # --- 4. Failure Analysis ---
    print("\n=== Failure Analysis ===")
    # Compute BCE loss per sample
    # Clip predictions to avoid log(0)
    epsilon = 1e-7
    preds_clipped = np.clip(avg_holdout_preds, epsilon, 1 - epsilon)

    # Binary Cross Entropy: - (y * log(p) + (1-y) * log(1-p))
    # Sum over classes for each sample
    sample_losses = -np.sum(
        y_holdout_true * np.log(preds_clipped)
        + (1 - y_holdout_true) * np.log(1 - preds_clipped),
        axis=1,
    )

    # Feature: Number of species present (Label Cardinality)
    num_species = np.sum(y_holdout_true, axis=1)

    # Correlation
    if np.std(sample_losses) > 0 and np.std(num_species) > 0:
        corr, _ = pearsonr(sample_losses, num_species)
        print(f"Correlation between Error (Loss) and Num Species: {corr:.4f}")
        if corr > 0.2:
            print(
                "Observation: Model struggles more with recordings containing multiple species."
            )
        elif corr < -0.2:
            print("Observation: Model struggles more with empty/sparse recordings.")
        else:
            print("Observation: Error is relatively independent of species count.")
    else:
        print("Correlation could not be computed (constant values).")

    # --- 5. Submission ---
    if final_metric > CONFIG["threshold"]:
        print("\n=== Generating Submission ===")

        test_ds = BirdDataset(
            df_test, CONFIG["input_dir"], image_size=CONFIG["image_size"], train=False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        test_preds_sum = np.zeros((len(df_test), num_classes))

        for path in model_paths:
            model = MILResNet18(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)

            preds = inference_fn(model, test_loader, device)
            test_preds_sum += preds

        avg_test_preds = test_preds_sum / CONFIG["num_folds"]

        # Format Submission
        submission_rows = []
        rec_ids = df_test["rec_id"].values

        for i, rec_id in enumerate(rec_ids):
            probs = avg_test_preds[i]
            for species_idx, prob in enumerate(probs):
                # ID format: rec_id * 100 + species_id
                row_id = int(rec_id * 100 + species_idx)
                submission_rows.append({"Id": row_id, "Probability": prob})

        sub_df = pd.DataFrame(submission_rows)
        sub_path = os.path.join(CONFIG["submission_dir"], "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {CONFIG['threshold']}. Submission skipped."
        )


if __name__ == "__main__":
    main()
