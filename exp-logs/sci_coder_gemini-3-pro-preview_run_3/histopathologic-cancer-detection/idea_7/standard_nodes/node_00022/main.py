import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import cv2

# Import library modules
from library.config import Config
from library.utils import seed_everything, load_checkpoint, calculate_metric
from library.dataset import load_dataset_metadata, PathologyDataset, get_transforms
from library.models import get_model
from library.engine import fit_model, predict_with_tta


def main():
    # --- 1. Initialization ---
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # --- 2. Data Loading & Splitting ---
    print("Loading training metadata...")
    df_train_full = load_dataset_metadata("train")

    # Create Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df_train_full["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["label"])
    ):
        df_train_full.loc[val_idx, "fold"] = fold

    # --- 3. Training Loop ---
    print("Starting Training...")

    # We only use one backbone now
    model_name = Config.MODEL_BACKBONES[0]
    print(f"\n=== Training Architecture: {model_name} ===")

    for fold in range(Config.N_FOLDS):
        print(f"  -- Fold {fold} --")

        # Split Data
        train_df = df_train_full[df_train_full["fold"] != fold].reset_index(drop=True)
        valid_df = df_train_full[df_train_full["fold"] == fold].reset_index(drop=True)

        # Datasets & Loaders
        train_ds = PathologyDataset(train_df, transform=get_transforms("train"))
        valid_ds = PathologyDataset(valid_df, transform=get_transforms("val"))

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model Setup
        model = get_model(model_name, pretrained=True, num_classes=1)
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Train
        save_path = os.path.join(Config.WORKING_DIR, f"{model_name}_fold_{fold}.pth")

        fit_model(
            model,
            train_loader,
            valid_loader,
            optimizer,
            criterion,
            device,
            save_path,
            epochs=Config.NUM_EPOCHS,
        )

        # Cleanup
        del (
            model,
            optimizer,
            criterion,
            train_loader,
            valid_loader,
            train_ds,
            valid_ds,
        )
        torch.cuda.empty_cache()

    # --- 5. Validation on Hold-Out Set ---
    print("\n=== Validating on Hold-Out Set ===")
    df_val = load_dataset_metadata("val")
    val_ds = PathologyDataset(df_val, transform=get_transforms("val"))
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Generate predictions from the 5 folds
    model_fold_preds = []
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"{model_name}_fold_{fold}.pth")
        model = get_model(model_name, num_classes=1)
        load_checkpoint(model_path, model, device=device)
        model = model.to(device)

        preds = predict_with_tta(model, val_loader, device)
        model_fold_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    # Average predictions (Ensemble)
    val_final_probs = np.mean(model_fold_preds, axis=0)
    val_targets = df_val["label"].values

    final_val_auc = calculate_metric(val_targets, val_final_probs)
    print(f"Final Validation Metric: {final_val_auc:.16f}")

    # --- 6. Failure Analysis ---
    print("\n=== Failure Analysis ===")
    errors = np.abs(val_targets - val_final_probs)

    print("Calculating metadata correlations on sample...")
    meta_stats = {"brightness": [], "contrast": [], "error": []}

    # Sample 1000 images for speed
    analysis_indices = np.random.choice(
        len(df_val), size=min(1000, len(df_val)), replace=False
    )

    for idx in analysis_indices:
        row = df_val.iloc[idx]
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(full_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            meta_stats["brightness"].append(np.mean(gray))
            meta_stats["contrast"].append(np.std(gray))
            meta_stats["error"].append(errors[idx])

    if len(meta_stats["error"]) > 0:
        df_analysis = pd.DataFrame(meta_stats)
        corr_b = df_analysis["brightness"].corr(df_analysis["error"])
        corr_c = df_analysis["contrast"].corr(df_analysis["error"])
        print(f"Error Correlation with Brightness: {corr_b:.4f}")
        print(f"Error Correlation with Contrast: {corr_c:.4f}")

    # --- 7. Test Inference & Submission ---
    THRESHOLD = 0.9946321378935362

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_val_auc} > {THRESHOLD}. Generating submission..."
        )

        df_test = load_dataset_metadata("test")
        test_ds = PathologyDataset(df_test, transform=get_transforms("test"))

        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Predicting test set with {model_name}...")
        test_fold_preds = []
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_fold_{fold}.pth"
            )
            model = get_model(model_name, num_classes=1)
            load_checkpoint(model_path, model, device=device)
            model = model.to(device)

            preds = predict_with_tta(model, test_loader, device)
            test_fold_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        # Average predictions
        test_final_probs = np.mean(test_fold_preds, axis=0)

        # Save
        submission = pd.DataFrame({"id": df_test["id"], "label": test_final_probs})

        submission.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")

    else:
        print(
            f"\nValidation metric {final_val_auc} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
