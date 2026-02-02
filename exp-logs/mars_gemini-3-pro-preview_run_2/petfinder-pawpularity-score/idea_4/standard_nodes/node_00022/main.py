import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Import library modules
import importlib
import library.config

importlib.reload(library.config)
from library.config import Config

# Verify patch (Cite debug_lesson_2)
if not hasattr(Config, "dense_features"):
    raise RuntimeError("Config patch failed! dense_features not found.")

from library.utils import seed_everything, get_rmse, unscale_target

import library.dataset

importlib.reload(library.dataset)
from library.dataset import PawpularityDataset, get_transforms

import library.model_factory

importlib.reload(library.model_factory)
from library.model_factory import PetModel

import library.trainer

importlib.reload(library.trainer)
from library.trainer import fit, predict, save_submission


def main():
    # 1. Setup
    seed_everything(Config.seed)

    # Use Config defaults (6 epochs)
    Config.debug = False

    # Ensure directories exist
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)

    print(f"Configuration:")
    print(f"  Device: {Config.device}")
    print(f"  Epochs per fold: {Config.epochs}")
    print(f"  Models: {list(Config.models.keys())}")

    # 2. Load Training Metadata
    train_metadata_path = Config.train_metadata_path
    if not os.path.exists(train_metadata_path):
        raise FileNotFoundError(f"Train metadata not found at {train_metadata_path}")

    full_train_df = pd.read_csv(train_metadata_path)

    # Prepare Stratified K-Fold
    # We stratify by binning the target to ensure distribution consistency
    num_bins = int(np.floor(1 + np.log2(len(full_train_df))))
    full_train_df["bins"] = pd.cut(
        full_train_df[Config.target_col], bins=num_bins, labels=False
    )

    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # 3. Train Base Models (Level 1)
    # We need to generate OOF predictions for the entire train set for each architecture

    for model_key, model_name in Config.models.items():
        print(f"\n{'='*40}")
        print(f"Training Architecture: {model_key} ({model_name})")
        print(f"{'='*40}")

        oof_preds_list = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_train_df, full_train_df["bins"])
        ):
            print(f"\n  Fold {fold+1}/{Config.n_folds}")

            # Split Data
            train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
            val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

            # Datasets
            train_dataset = PawpularityDataset(
                train_df, transforms=get_transforms("train")
            )
            val_dataset = PawpularityDataset(val_df, transforms=get_transforms("valid"))

            # Loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.batch_size,
                shuffle=True,
                num_workers=Config.num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.batch_size,
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )

            # Model
            model = PetModel(model_name=model_key, pretrained=True)
            model.to(Config.device)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": Config.backbone_lr},
                    {"params": model.head.parameters(), "lr": Config.head_lr},
                ],
                weight_decay=Config.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.T_max, eta_min=Config.min_lr
            )

            # Train
            save_path = Config().get_model_path(model_key, fold)
            best_val_preds_scaled = fit(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                Config.device,
                epochs=Config.epochs,
                save_path=save_path,
                patience=Config.patience,
            )

            # Store OOF predictions
            # fit returns scaled predictions [0, 1], which is what StackingTrainer expects
            val_ids = val_df["Id"].values
            fold_oof_df = pd.DataFrame(
                {"Id": val_ids, Config.target_col: best_val_preds_scaled}
            )
            oof_preds_list.append(fold_oof_df)

            # Cleanup
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

        # Concatenate all folds for this architecture
        full_oof_df = pd.concat(oof_preds_list, ignore_index=True)

        # Save OOF file
        oof_path = Config().get_oof_path(model_key)
        full_oof_df.to_csv(oof_path, index=False)
        print(f"Saved OOF predictions for {model_key} to {oof_path}")

    # 4. Validation on Hold-out Set
    print(f"\n{'='*40}")
    print("Evaluating on Hold-out Validation Set")
    print(f"{'='*40}")

    val_metadata_path = Config.val_metadata_path
    val_df = pd.read_csv(val_metadata_path)

    val_dataset = PawpularityDataset(val_df, transforms=get_transforms("valid"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Collect predictions from all base models
    model_preds_map = {}  # key -> list of preds from 5 folds

    for model_key in Config.models.keys():
        fold_preds = []
        for fold in range(Config.n_folds):
            model_path = Config().get_model_path(model_key, fold)
            model = PetModel(model_name=model_key, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=Config.device))
            model.to(Config.device)

            # predict() returns unscaled [1, 100]
            preds_unscaled = predict(model, val_loader, Config.device)
            fold_preds.append(preds_unscaled)

            del model
            torch.cuda.empty_cache()

        # Average across folds for this architecture
        avg_preds = np.mean(fold_preds, axis=0)
        model_preds_map[model_key] = avg_preds

    # Simple Average Ensemble
    final_val_preds = np.mean(list(model_preds_map.values()), axis=0)

    # Calculate RMSE
    val_targets = val_df[Config.target_col].values
    final_rmse = get_rmse(val_targets, final_val_preds)

    print(f"Final Validation Metric: {final_rmse}")

    # 6. Failure Analysis
    print(f"\n{'='*40}")
    print("Failure Analysis")
    print(f"{'='*40}")

    val_df["pred"] = final_val_preds
    val_df["error"] = np.abs(val_df["Pawpularity"] - val_df["pred"])

    # Select numeric columns for correlation (excluding Id, file_path, Pawpularity, pred, error)
    feature_cols = [
        c
        for c in val_df.columns
        if c not in ["Id", "file_path", "Pawpularity", "pred", "error"]
    ]

    print("Correlation between Error and Features:")
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(val_df[col]):
            corr = val_df["error"].corr(val_df[col])
            print(f"  {col}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 17.184850648081728
    if final_rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({final_rmse}) < Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_metadata_path = Config.test_metadata_path
        test_df = pd.read_csv(test_metadata_path)

        test_dataset = PawpularityDataset(
            test_df, transforms=get_transforms("test"), test=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        test_preds_map = {}

        for model_key in Config.models.keys():
            fold_preds = []
            for fold in range(Config.n_folds):
                model_path = Config().get_model_path(model_key, fold)
                model = PetModel(model_name=model_key, pretrained=False)
                model.load_state_dict(
                    torch.load(model_path, map_location=Config.device)
                )
                model.to(Config.device)

                preds_unscaled = predict(model, test_loader, Config.device)
                fold_preds.append(preds_unscaled)

                del model
                torch.cuda.empty_cache()

            avg_preds = np.mean(fold_preds, axis=0)
            test_preds_map[model_key] = avg_preds

        final_test_preds = np.mean(list(test_preds_map.values()), axis=0)

        save_submission(test_df["Id"].values, final_test_preds, Config.submission_path)

    else:
        print(
            f"\nValidation RMSE ({final_rmse}) >= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
