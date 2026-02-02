import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    get_pos_weight,
    save_oof_preds,
    load_oof_preds,
)
from library.loss import BornAgainLoss
from library.models import BirdModel
from library.dataset import BirdDataset, get_transforms
from library.engine import train_one_epoch, validate, predict

# --- Configuration Overrides for Fast Baseline ---
# Adjusting parameters to fit within the time limit while maintaining the strategy structure.
Config.EPOCHS = 5
Config.NUM_FOLDS = 1  # Using the single split provided by metadata
Config.BATCH_SIZE = 32


def run_training(model_name, train_loader, device, pos_weight, soft_targets_path=None):
    """
    Trains a single model for Config.EPOCHS.
    """
    model = BirdModel(model_name, pretrained=True).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss function handles hard labels and optional soft labels
    loss_fn = BornAgainLoss(
        pos_weight=pos_weight, distillation_lambda=Config.DISTILLATION_LAMBDA
    )

    # Training Loop
    for epoch in range(Config.EPOCHS):
        _ = train_one_epoch(model, train_loader, optimizer, device, epoch, loss_fn)

    return model


def get_predictions(model, loader, device):
    """
    Runs inference to get probabilities.
    """
    ids, preds = predict(model, loader, device)
    return ids, preds


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Calculate Class Weights
    pos_weight = get_pos_weight(df_train, device)

    # Transforms
    train_transforms = get_transforms(mode="train")
    val_transforms = get_transforms(mode="val")

    # =========================================================================
    # Generation 0: Anchors
    # =========================================================================
    # Train ResNet18 and EfficientNet-B0 on Hard Labels
    gen0_models = []
    gen0_preds_train = []

    # Dataset for Gen 0 (No soft labels)
    ds_train_gen0 = BirdDataset(
        df_train, transforms=train_transforms, soft_labels_path=None
    )
    dl_train_gen0 = DataLoader(
        ds_train_gen0,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    # Dataset for inference on training set (to generate soft labels for next gen)
    # Using val_transforms for deterministic prediction
    ds_train_infer = BirdDataset(
        df_train, transforms=val_transforms, soft_labels_path=None
    )
    dl_train_infer = DataLoader(
        ds_train_infer,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    for model_name in Config.ANCHOR_MODELS:
        model = run_training(
            model_name, dl_train_gen0, device, pos_weight, soft_targets_path=None
        )
        gen0_models.append(model)

        # Generate Soft Targets (Predict on Training Set)
        _, preds = get_predictions(model, dl_train_infer, device)
        gen0_preds_train.append(preds)

    # Average Soft Targets
    avg_gen0_preds = np.mean(gen0_preds_train, axis=0)

    # Save Gen 0 Soft Targets
    gen0_soft_path = os.path.join(Config.GEN0_DIR, "soft_targets_train.parquet")
    save_oof_preds(avg_gen0_preds, df_train["rec_id"].values, gen0_soft_path)

    # =========================================================================
    # Generation 1: Stabilization
    # =========================================================================
    # Train Full Ensemble using Gen 0 Soft Targets
    gen1_models = []
    gen1_preds_train = []

    # Dataset for Gen 1 (With Gen 0 soft labels)
    ds_train_gen1 = BirdDataset(
        df_train, transforms=train_transforms, soft_labels_path=gen0_soft_path
    )
    dl_train_gen1 = DataLoader(
        ds_train_gen1,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    for model_name in Config.FULL_ENSEMBLE_MODELS:
        model = run_training(
            model_name,
            dl_train_gen1,
            device,
            pos_weight,
            soft_targets_path=gen0_soft_path,
        )
        gen1_models.append(model)

        # Generate Soft Targets
        _, preds = get_predictions(model, dl_train_infer, device)
        gen1_preds_train.append(preds)

    avg_gen1_preds = np.mean(gen1_preds_train, axis=0)
    gen1_soft_path = os.path.join(Config.GEN1_DIR, "soft_targets_train.parquet")
    save_oof_preds(avg_gen1_preds, df_train["rec_id"].values, gen1_soft_path)

    # =========================================================================
    # Generation 2: Refinement
    # =========================================================================
    # Train Full Ensemble using Gen 1 Soft Targets
    gen2_models = []

    ds_train_gen2 = BirdDataset(
        df_train, transforms=train_transforms, soft_labels_path=gen1_soft_path
    )
    dl_train_gen2 = DataLoader(
        ds_train_gen2,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    for model_name in Config.FULL_ENSEMBLE_MODELS:
        model = run_training(
            model_name,
            dl_train_gen2,
            device,
            pos_weight,
            soft_targets_path=gen1_soft_path,
        )
        gen2_models.append(model)

    # =========================================================================
    # Evaluation
    # =========================================================================
    # Validation Dataset
    ds_val = BirdDataset(df_val, transforms=val_transforms)
    dl_val = DataLoader(
        ds_val,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Ensemble Prediction on Validation Set
    val_preds_list = []

    # Get targets once
    all_val_targets = []
    for batch in dl_val:
        all_val_targets.append(batch["targets"].numpy())
    val_targets = np.concatenate(all_val_targets, axis=0)

    for model in gen2_models:
        _, preds = get_predictions(model, dl_val, device)
        val_preds_list.append(preds)

    avg_val_preds = np.mean(val_preds_list, axis=0)

    # Calculate Metric
    # Cite debug_lesson_6
    class_aucs = []
    for i in range(val_targets.shape[1]):
        # Only calculate AUC if the class has both positive and negative samples
        if len(np.unique(val_targets[:, i])) > 1:
            class_aucs.append(roc_auc_score(val_targets[:, i], avg_val_preds[:, i]))

    if len(class_aucs) > 0:
        final_metric = np.mean(class_aucs)
    else:
        final_metric = 0.5

    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # Failure Analysis
    # =========================================================================
    # Calculate per-sample error (Mean Absolute Error across classes)
    per_sample_error = np.mean(np.abs(val_targets - avg_val_preds), axis=1)

    # Feature: Image Mean Intensity
    img_means = []
    for idx in range(len(df_val)):
        path = os.path.join(Config.INPUT_DIR, df_val.iloc[idx]["file_path_spec"])
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            img_means.append(np.mean(img))
        else:
            img_means.append(0)

    correlation, _ = pearsonr(per_sample_error, img_means)
    print(
        f"Failure Analysis: Correlation between Error and Image Intensity: {correlation:.4f}"
    )

    # =========================================================================
    # Submission
    # =========================================================================
    THRESHOLD = 0.92133638985917

    if final_metric > THRESHOLD:
        # Cyclic TTA
        test_preds_accum = []

        for shift in Config.TTA_SHIFTS:
            # Create TTA dataset
            ds_test_tta = BirdDataset(
                df_test, transforms=get_transforms(mode="test", tta_shift=shift)
            )
            dl_test_tta = DataLoader(
                ds_test_tta,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            for model in gen2_models:
                ids, preds = get_predictions(model, dl_test_tta, device)
                test_preds_accum.append(preds)

        # Average all TTA predictions from all models
        final_test_preds = np.mean(test_preds_accum, axis=0)

        # Format Submission
        submission_rows = []
        rec_ids = df_test["rec_id"].values

        for i, rec_id in enumerate(rec_ids):
            probs = final_test_preds[i]
            for species_idx, prob in enumerate(probs):
                # Id mapping: rec_id * 100 + species_id
                row_id = int(rec_id * 100 + species_idx)
                submission_rows.append({"Id": row_id, "Probability": prob})

        df_sub = pd.DataFrame(submission_rows)
        df_sub.to_csv(
            os.path.join(Config.SUBMISSION_DIR, "submission.csv"), index=False
        )


if __name__ == "__main__":
    main()
