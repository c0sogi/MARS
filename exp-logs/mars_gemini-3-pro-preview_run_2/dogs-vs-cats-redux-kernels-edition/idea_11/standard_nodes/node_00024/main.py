import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import cv2

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.dataset import CatDogDataset, get_transforms
from library.models import get_model
from library.engine import train_model, inference_fn

# =============================================================================
# Configuration Overrides for Fast Baseline
# =============================================================================
# To ensure execution within the time limit while verifying the pipeline logic.
Config.N_FOLDS = 2
Config.EPOCHS = 3
Config.BATCH_SIZE = 32
Config.NUM_WORKERS = 4


def run_training():
    # 1. Setup
    Config.create_directories()
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_holdout_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Store OOF predictions and Test predictions for each model
    # Structure: stage -> model -> 'oof': [], 'test': [], 'holdout': []
    results = {"stage1": {}, "stage2": {}}

    # Define Folds on train_df
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # =========================================================================
    # STAGE 1: Supervised Learning on Labeled Data
    # =========================================================================
    print("\n=== STAGE 1: Supervised Learning ===")

    for model_name in Config.MODEL_NAMES:
        print(f"\nTraining Model: {model_name}")
        results["stage1"][model_name] = {
            "oof": np.zeros(len(train_df)),
            "test": np.zeros(len(test_df)),
            "holdout": np.zeros(len(val_holdout_df)),
        }

        test_preds_accum = np.zeros(len(test_df))
        holdout_preds_accum = np.zeros(len(val_holdout_df))

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df["label"])
        ):
            print(f"  Fold {fold+1}/{Config.N_FOLDS}")

            # Split Data
            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

            # Datasets & Loaders
            train_ds = CatDogDataset(fold_train_df, transforms=get_transforms("train"))
            val_ds = CatDogDataset(fold_val_df, transforms=get_transforms("valid"))

            train_loader = DataLoader(
                train_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Model, Optimizer, Scheduler
            model = get_model(model_name, pretrained=True)
            model = model.to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )

            # Train
            save_path = os.path.join(
                Config.CHECKPOINT_DIR, f"stage1_{model_name}_fold_{fold}.pth"
            )
            train_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                Config.EPOCHS,
                patience=Config.EPOCHS,
                save_path=save_path,
            )

            # Load Best & Inference
            model.load_state_dict(torch.load(save_path, map_location=device))

            # OOF Inference
            val_oof_preds = inference_fn(model, val_loader, device)
            results["stage1"][model_name]["oof"][val_idx] = val_oof_preds

            # Test Inference
            test_ds = CatDogDataset(test_df, transforms=get_transforms("valid"))
            test_loader = DataLoader(
                test_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            test_preds_accum += inference_fn(model, test_loader, device)

            # Holdout Inference
            holdout_ds = CatDogDataset(
                val_holdout_df, transforms=get_transforms("valid")
            )
            holdout_loader = DataLoader(
                holdout_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            holdout_preds_accum += inference_fn(model, holdout_loader, device)

            # Cleanup
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

        # Average predictions
        results["stage1"][model_name]["test"] = test_preds_accum / Config.N_FOLDS
        results["stage1"][model_name]["holdout"] = holdout_preds_accum / Config.N_FOLDS

    # =========================================================================
    # PSEUDO-LABELING
    # =========================================================================
    print("\n=== Generating Pseudo-Labels ===")
    # Average predictions from all Stage 1 models
    avg_test_preds = np.zeros(len(test_df))
    for model_name in Config.MODEL_NAMES:
        avg_test_preds += results["stage1"][model_name]["test"]
    avg_test_preds /= len(Config.MODEL_NAMES)

    # Create Pseudo-labeled DataFrame
    pseudo_df = test_df.copy()
    pseudo_df["label"] = avg_test_preds
    if "id" in pseudo_df.columns:
        pseudo_df = pseudo_df.drop(columns=["id"])

    print(f"Generated pseudo-labels for {len(pseudo_df)} test samples.")

    # =========================================================================
    # STAGE 2: Semi-Supervised Learning
    # =========================================================================
    print("\n=== STAGE 2: Semi-Supervised Learning ===")

    for model_name in Config.MODEL_NAMES:
        print(f"\nTraining Model (Stage 2): {model_name}")
        results["stage2"][model_name] = {
            "oof": np.zeros(len(train_df)),
            "test": np.zeros(len(test_df)),
            "holdout": np.zeros(len(val_holdout_df)),
        }

        test_preds_accum = np.zeros(len(test_df))
        holdout_preds_accum = np.zeros(len(val_holdout_df))

        # Re-use the same folds on train_df to ensure OOF integrity
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df["label"])
        ):
            print(f"  Fold {fold+1}/{Config.N_FOLDS}")

            # Original Labeled Split
            fold_train_labeled = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

            # Combine Labeled Train + Pseudo-Labeled Test
            fold_train_combined = pd.concat(
                [fold_train_labeled, pseudo_df], axis=0
            ).reset_index(drop=True)

            # Datasets & Loaders
            train_ds = CatDogDataset(
                fold_train_combined, transforms=get_transforms("train")
            )
            val_ds = CatDogDataset(fold_val_df, transforms=get_transforms("valid"))

            train_loader = DataLoader(
                train_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Model (Re-initialize)
            model = get_model(model_name, pretrained=True)
            model = model.to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )

            # Train
            save_path = os.path.join(
                Config.CHECKPOINT_DIR, f"stage2_{model_name}_fold_{fold}.pth"
            )
            train_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                Config.EPOCHS,
                patience=Config.EPOCHS,
                save_path=save_path,
            )

            # Load Best & Inference
            model.load_state_dict(torch.load(save_path, map_location=device))

            # OOF Inference (on original labeled validation fold)
            val_oof_preds = inference_fn(model, val_loader, device)
            results["stage2"][model_name]["oof"][val_idx] = val_oof_preds

            # Test Inference
            test_ds = CatDogDataset(test_df, transforms=get_transforms("valid"))
            test_loader = DataLoader(
                test_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            test_preds_accum += inference_fn(model, test_loader, device)

            # Holdout Inference
            holdout_ds = CatDogDataset(
                val_holdout_df, transforms=get_transforms("valid")
            )
            holdout_loader = DataLoader(
                holdout_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )
            holdout_preds_accum += inference_fn(model, holdout_loader, device)

            # Cleanup
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

        # Average predictions
        results["stage2"][model_name]["test"] = test_preds_accum / Config.N_FOLDS
        results["stage2"][model_name]["holdout"] = holdout_preds_accum / Config.N_FOLDS

    # =========================================================================
    # STACKING
    # =========================================================================
    print("\n=== Stacking Ensemble ===")

    # Prepare Data for Meta-Learner
    X_train_meta = []
    for model_name in Config.MODEL_NAMES:
        X_train_meta.append(results["stage2"][model_name]["oof"])
    X_train_meta = np.column_stack(X_train_meta)
    y_train_meta = train_df["label"].values

    # Train Logistic Regression
    meta_model = LogisticRegression(random_state=Config.SEED)
    meta_model.fit(X_train_meta, y_train_meta)
    print(f"Meta-Learner Coefficients: {meta_model.coef_}")

    # Prepare Test Data for Meta-Learner
    X_test_meta = []
    for model_name in Config.MODEL_NAMES:
        X_test_meta.append(results["stage2"][model_name]["test"])
    X_test_meta = np.column_stack(X_test_meta)

    # Prepare Holdout Data for Meta-Learner
    X_holdout_meta = []
    for model_name in Config.MODEL_NAMES:
        X_holdout_meta.append(results["stage2"][model_name]["holdout"])
    X_holdout_meta = np.column_stack(X_holdout_meta)

    # Final Predictions
    final_test_preds = meta_model.predict_proba(X_test_meta)[:, 1]
    final_holdout_preds = meta_model.predict_proba(X_holdout_meta)[:, 1]

    # =========================================================================
    # VALIDATION & FAILURE ANALYSIS
    # =========================================================================
    print("\n=== Validation & Failure Analysis ===")

    y_true_holdout = val_holdout_df["label"].values
    final_metric = calculate_log_loss(y_true_holdout, final_holdout_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    residuals = np.abs(y_true_holdout - final_holdout_preds)

    print("Performing failure analysis on holdout set...")
    meta_features = []
    for idx, row in val_holdout_df.iterrows():
        filepath = os.path.join(Config.INPUT_DIR, row["filepath"])
        img = cv2.imread(filepath)
        if img is not None:
            h, w, _ = img.shape
            meta_features.append(
                {
                    "width": w,
                    "height": h,
                    "aspect_ratio": w / h,
                    "residual": residuals[idx],
                }
            )

    if meta_features:
        meta_df = pd.DataFrame(meta_features)
        print("\nCorrelation between Error (Residual) and Image Features:")
        for feat in ["width", "height", "aspect_ratio"]:
            corr = meta_df[feat].corr(meta_df["residual"])
            print(f"  {feat}: {corr:.4f}")

    # =========================================================================
    # SUBMISSION
    # =========================================================================
    THRESHOLD = 0.01366509944361823

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        submission_df = pd.DataFrame({"id": test_df["id"], "label": final_test_preds})
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run_training()
