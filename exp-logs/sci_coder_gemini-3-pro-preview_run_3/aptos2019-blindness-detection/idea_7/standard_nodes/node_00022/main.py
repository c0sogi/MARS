import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
import cv2

# Import from library
from library.config import Config
from library.data import (
    process_and_cache_data,
    RetinaDataset,
    get_transforms,
    get_test_loader,
)
from library.model import DRModel
from library.utils import (
    seed_everything,
    quadratic_weighted_kappa,
    save_checkpoint,
    load_checkpoint,
)
from library.engine import train_one_epoch, validate, inference


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    # Optimization: Adjust epochs for fast baseline execution within time limits
    # Stage 1: Fast structure learning
    Config.STAGE1_EPOCHS = 6
    # Stage 2: High-res fine-tuning
    Config.STAGE2_EPOCHS = 4

    print("Loading training metadata...")
    full_train_df = pd.read_csv(Config.TRAIN_CSV)

    # 2. 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_train_df, full_train_df["diagnosis"])
    ):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Create Fold DataFrames
        train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

        # -------------------------------------------------------
        # Stage 1: Structure Learning (512x512)
        # -------------------------------------------------------
        print(f"--- Stage 1: 512x512 Training ---")

        # Process/Cache Data
        train_imgs_s1, train_labels_s1 = process_and_cache_data(
            train_df,
            Config.STAGE1_IMG_SIZE,
            f"fold_{fold}_train",
            load_cached_data=True,
        )
        val_imgs_s1, val_labels_s1 = process_and_cache_data(
            val_df, Config.STAGE1_IMG_SIZE, f"fold_{fold}_val", load_cached_data=True
        )

        # Create Loaders
        train_ds_s1 = RetinaDataset(
            train_imgs_s1, train_labels_s1, transform=get_transforms("train")
        )
        val_ds_s1 = RetinaDataset(
            val_imgs_s1, val_labels_s1, transform=get_transforms("val")
        )

        train_loader_s1 = torch.utils.data.DataLoader(
            train_ds_s1,
            batch_size=Config.STAGE1_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader_s1 = torch.utils.data.DataLoader(
            val_ds_s1,
            batch_size=Config.STAGE1_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = DRModel(Config.BACKBONE, pretrained=True)
        model.to(Config.DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=Config.STAGE1_LR)
        criterion = nn.MSELoss()

        # Train Stage 1
        for epoch in range(Config.STAGE1_EPOCHS):
            avg_loss = train_one_epoch(
                model,
                train_loader_s1,
                criterion,
                optimizer,
                Config.DEVICE,
                accumulation_steps=Config.STAGE1_GRAD_ACCUM,
            )
            # Optional: Validate to check progress, but we rely on Stage 2 for model selection
            # val_loss, val_qwk = validate(model, val_loader_s1, criterion, Config.DEVICE)
            # print(f"Stage 1 Epoch {epoch+1}: Loss={avg_loss:.4f}")

        # Clean up Stage 1 data to free memory
        del (
            train_imgs_s1,
            val_imgs_s1,
            train_ds_s1,
            val_ds_s1,
            train_loader_s1,
            val_loader_s1,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # -------------------------------------------------------
        # Stage 2: Fine-Grained Adaptation (1024x1024)
        # -------------------------------------------------------
        print(f"--- Stage 2: 1024x1024 Fine-Tuning ---")

        # Process/Cache Data
        train_imgs_s2, train_labels_s2 = process_and_cache_data(
            train_df,
            Config.STAGE2_IMG_SIZE,
            f"fold_{fold}_train",
            load_cached_data=True,
        )
        val_imgs_s2, val_labels_s2 = process_and_cache_data(
            val_df, Config.STAGE2_IMG_SIZE, f"fold_{fold}_val", load_cached_data=True
        )

        train_ds_s2 = RetinaDataset(
            train_imgs_s2, train_labels_s2, transform=get_transforms("train")
        )
        val_ds_s2 = RetinaDataset(
            val_imgs_s2, val_labels_s2, transform=get_transforms("val")
        )

        # Batch size is small (2), so we use gradient accumulation
        train_loader_s2 = torch.utils.data.DataLoader(
            train_ds_s2,
            batch_size=Config.STAGE2_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader_s2 = torch.utils.data.DataLoader(
            val_ds_s2,
            batch_size=Config.STAGE2_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Update Optimizer for Fine-tuning
        optimizer = optim.AdamW(model.parameters(), lr=Config.STAGE2_LR)

        # Enable Gradient Checkpointing
        model.set_grad_checkpointing(enable=True)

        best_fold_qwk = -np.inf

        for epoch in range(Config.STAGE2_EPOCHS):
            avg_loss = train_one_epoch(
                model,
                train_loader_s2,
                criterion,
                optimizer,
                Config.DEVICE,
                accumulation_steps=Config.STAGE2_GRAD_ACCUM,
            )
            val_loss, val_qwk = validate(model, val_loader_s2, criterion, Config.DEVICE)
            print(
                f"Stage 2 Epoch {epoch+1}: Loss={avg_loss:.4f}, Val QWK={val_qwk:.4f}"
            )

            if val_qwk > best_fold_qwk:
                best_fold_qwk = val_qwk
                save_path = os.path.join(Config.MODEL_DIR, f"model_fold_{fold}.pth")
                save_checkpoint(model, optimizer, epoch, val_qwk, save_path)

        fold_scores.append(best_fold_qwk)

        # Cleanup Fold
        del (
            model,
            optimizer,
            train_imgs_s2,
            val_imgs_s2,
            train_ds_s2,
            val_ds_s2,
            train_loader_s2,
            val_loader_s2,
        )
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nCross-Validation Scores: {fold_scores}")
    print(f"Average CV Score: {np.mean(fold_scores):.4f}")

    # 3. Validation on Hold-out Set & Failure Analysis
    print("\nStarting Validation on Hold-out Set...")

    # Load Hold-out Data
    holdout_df = pd.read_csv(Config.VAL_CSV)

    # Process for Inference (1024x1024)
    val_imgs, _ = process_and_cache_data(
        holdout_df, Config.STAGE2_IMG_SIZE, "holdout_val", load_cached_data=True
    )

    # Create Inference Loader
    val_ds = RetinaDataset(val_imgs, labels=None, transform=get_transforms("test"))
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensemble Inference
    ensemble_preds = np.zeros(len(holdout_df))
    models_loaded = 0

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.MODEL_DIR, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            continue

        model = DRModel(Config.BACKBONE, pretrained=False)
        model.to(Config.DEVICE)
        load_checkpoint(model, model_path, device=Config.DEVICE)

        preds = inference(model, val_loader, Config.DEVICE)
        ensemble_preds += preds
        models_loaded += 1

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if models_loaded > 0:
        ensemble_preds /= models_loaded

    # Calculate Final Metric
    final_preds_int = np.round(ensemble_preds.clip(0, 4)).astype(int)
    final_score = quadratic_weighted_kappa(
        holdout_df["diagnosis"].values, final_preds_int
    )

    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    residuals = np.abs(holdout_df["diagnosis"].values - ensemble_preds)

    # Collect Meta-features
    widths, heights, intensities = [], [], []

    for _, row in holdout_df.iterrows():
        fpath = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            img = cv2.imread(fpath)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                # Quick intensity calc
                intensities.append(img.mean() / 255.0)
            else:
                widths.append(0)
                heights.append(0)
                intensities.append(0)
        except:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    analysis_df = pd.DataFrame(
        {
            "residual": residuals,
            "width": widths,
            "height": heights,
            "intensity": intensities,
        }
    )

    print("Correlation between Error and Meta-features:")
    for feat in ["width", "height", "intensity"]:
        corr, _ = spearmanr(analysis_df["residual"], analysis_df[feat])
        print(f"  {feat}: {corr:.4f}")

    # 4. Submission
    THRESHOLD = 0.9241120634346159
    if final_score > THRESHOLD:
        print("\nGenerating Submission...")

        test_loader, test_df = get_test_loader(
            Config.INFERENCE_IMG_SIZE,
            Config.INFERENCE_BATCH_SIZE,
            load_cached_data=True,
        )

        test_ensemble_preds = np.zeros(len(test_df))
        models_loaded = 0

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.MODEL_DIR, f"model_fold_{fold}.pth")
            if not os.path.exists(model_path):
                continue

            model = DRModel(Config.BACKBONE, pretrained=False)
            model.to(Config.DEVICE)
            load_checkpoint(model, model_path, device=Config.DEVICE)

            preds = inference(model, test_loader, Config.DEVICE)
            test_ensemble_preds += preds
            models_loaded += 1

            del model
            torch.cuda.empty_cache()
            gc.collect()

        if models_loaded > 0:
            test_ensemble_preds /= models_loaded

        final_test_preds = np.round(test_ensemble_preds.clip(0, 4)).astype(int)

        submission = pd.DataFrame(
            {"id_code": test_df["id_code"], "diagnosis": final_test_preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation score {final_score} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
