import sys
import os
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratifiedKFold

# Import library modules
from library.config import CFG
from library.utils import set_seed, get_score, get_logger
from library.dataset import BirdDataset
from library.models import BirdModel
from library.engine import train_fn, valid_fn
from library.sam import SAM


def main():
    # 1. Setup
    set_seed(CFG.seed)
    # Ensure output directories exist (handled by CFG.setup, but good to be sure)
    os.makedirs(CFG.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(CFG.output_dir, "train.log"))

    # Optimization for fast baseline execution
    # 206 training samples / 32 batch_size ~= 7 steps per epoch.
    # 50 epochs * 7 steps = 350 steps per model.
    # 15 models * 350 steps = 5250 total steps.
    # With SAM (2x forward/backward), this fits comfortably within the time limit.
    CFG.epochs = 50

    logger.info(f"Starting run with {CFG.epochs} epochs per model.")

    # 2. Load Metadata
    df_train_full = pd.read_csv(CFG.train_metadata_path)
    df_val_holdout = pd.read_csv(CFG.val_metadata_path)
    df_test = pd.read_csv(CFG.test_metadata_path)

    # 3. Prepare Cross-Validation
    # We use IterativeStratifiedKFold to maintain multi-label distribution across folds
    X = df_train_full["rec_id"].values.reshape(-1, 1)
    label_cols = [c for c in df_train_full.columns if c.startswith("species_")]
    y = df_train_full[label_cols].values

    kfold = IterativeStratifiedKFold(n_splits=CFG.n_folds, order=1)

    folds = []
    for train_idx, val_idx in kfold.split(X, y):
        folds.append((train_idx, val_idx))

    # 4. Training Loop
    model_paths = []

    for backbone in CFG.backbones:
        logger.info(f"=== Training Backbone: {backbone} ===")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"  Fold {fold_idx+1}/{CFG.n_folds}")

            # Split data for this fold
            df_train_fold = df_train_full.iloc[train_idx].reset_index(drop=True)
            df_valid_fold = df_train_full.iloc[val_idx].reset_index(drop=True)

            # Initialize Datasets
            train_dataset = BirdDataset(df_train_fold, mode="train")
            valid_dataset = BirdDataset(df_valid_fold, mode="val")

            # Initialize DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=CFG.batch_size,
                shuffle=True,
                num_workers=CFG.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers,
                pin_memory=True,
            )

            # Calculate positive weights for BCEWithLogitsLoss
            fold_labels = df_train_fold[label_cols].values
            num_pos = fold_labels.sum(axis=0)
            num_neg = len(fold_labels) - num_pos
            # Handle potential division by zero if a class is absent in a fold (unlikely with StratifiedKFold but possible)
            pos_weight = torch.tensor(
                np.where(num_pos > 0, num_neg / num_pos, 1.0), dtype=torch.float32
            ).to(CFG.device)

            # Initialize Model
            model = BirdModel(backbone_name=backbone, pretrained=True)
            model.to(CFG.device)

            # Loss Function
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # Optimizer: SAM wrapping AdamW
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(
                model.parameters(),
                base_optimizer,
                rho=CFG.sam_rho,
                lr=CFG.lr,
                weight_decay=CFG.weight_decay,
            )

            # Training Loop
            best_score = -1
            best_model_path = os.path.join(
                CFG.output_dir, f"{backbone}_fold{fold_idx}.pth"
            )

            for epoch in range(CFG.epochs):
                avg_loss = train_fn(
                    train_loader, model, criterion, optimizer, CFG.device
                )
                val_loss, val_score, _ = valid_fn(
                    valid_loader, model, criterion, CFG.device
                )

                if val_score > best_score:
                    best_score = val_score
                    torch.save(model.state_dict(), best_model_path)

            model_paths.append((backbone, best_model_path))
            logger.info(f"    Best Score: {best_score:.4f}")

            # Cleanup to free GPU memory
            del (
                model,
                optimizer,
                criterion,
                train_loader,
                valid_loader,
                train_dataset,
                valid_dataset,
            )
            torch.cuda.empty_cache()
            gc.collect()

    # 5. Inference & Ensemble on Hold-out Validation Set
    logger.info("=== Validating on Hold-out Set ===")

    val_preds_accum = np.zeros((len(df_val_holdout), CFG.num_classes))
    total_ensembles = len(model_paths) * len(CFG.tta_shifts)

    for backbone, path in model_paths:
        # Load model
        model = BirdModel(backbone_name=backbone, pretrained=False)
        model.load_state_dict(torch.load(path, map_location=CFG.device))
        model.to(CFG.device)
        model.eval()

        # Test-Time Augmentation (TTA)
        for shift in CFG.tta_shifts:
            ds = BirdDataset(df_val_holdout, mode="val", tta_shift=shift)
            dl = DataLoader(
                ds,
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers,
            )

            preds = []
            with torch.no_grad():
                for images, _ in dl:
                    images = images.to(CFG.device)
                    logits = model(images)
                    preds.append(torch.sigmoid(logits).cpu().numpy())

            val_preds_accum += np.concatenate(preds)

        del model, ds, dl
        torch.cuda.empty_cache()
        gc.collect()

    val_preds_avg = val_preds_accum / total_ensembles
    val_targets = df_val_holdout[label_cols].values

    final_val_metric = get_score(val_targets, val_preds_avg)
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    logger.info("=== Failure Analysis ===")

    # Calculate Mean Absolute Error per sample
    errors = np.abs(val_targets - val_preds_avg).mean(axis=1)

    # Feature 1: Number of Labels (Cardinality)
    num_labels = val_targets.sum(axis=1)
    corr_labels = np.corrcoef(errors, num_labels)[0, 1]
    print(f"Correlation (Error vs Num Labels): {corr_labels:.4f}")

    # Feature 2: Image Brightness (Input Feature)
    brightness = []
    for _, row in df_val_holdout.iterrows():
        # Reconstruct path logic from dataset.py
        original_spec_path = row["file_path_spec"]
        filename = os.path.basename(original_spec_path)
        full_path = os.path.join(CFG.filtered_spectrogram_dir, filename)

        if not os.path.exists(full_path):
            full_path = os.path.join(
                "./input", "supplemental_data", "filtered_spectrograms", filename
            )

        if os.path.exists(full_path):
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                brightness.append(np.mean(img))
            else:
                brightness.append(0)
        else:
            brightness.append(0)

    if len(brightness) == len(errors):
        corr_bright = np.corrcoef(errors, brightness)[0, 1]
        print(f"Correlation (Error vs Image Brightness): {corr_bright:.4f}")

    # 7. Submission
    threshold = 0.9167709334579945

    if final_val_metric > threshold:
        logger.info("=== Generating Submission ===")

        test_preds_accum = np.zeros((len(df_test), CFG.num_classes))

        for backbone, path in model_paths:
            model = BirdModel(backbone_name=backbone, pretrained=False)
            model.load_state_dict(torch.load(path, map_location=CFG.device))
            model.to(CFG.device)
            model.eval()

            for shift in CFG.tta_shifts:
                ds = BirdDataset(df_test, mode="test", tta_shift=shift)
                dl = DataLoader(
                    ds,
                    batch_size=CFG.batch_size,
                    shuffle=False,
                    num_workers=CFG.num_workers,
                )

                preds = []
                with torch.no_grad():
                    for images, _ in dl:
                        images = images.to(CFG.device)
                        logits = model(images)
                        preds.append(torch.sigmoid(logits).cpu().numpy())

                test_preds_accum += np.concatenate(preds)

            del model, ds, dl
            torch.cuda.empty_cache()
            gc.collect()

        test_preds_avg = test_preds_accum / total_ensembles

        # Format Submission
        submission_rows = []
        rec_ids = df_test["rec_id"].values

        for i, rec_id in enumerate(rec_ids):
            probs = test_preds_avg[i]
            for species_idx, prob in enumerate(probs):
                # Id format: rec_id * 100 + species_id
                row_id = int(rec_id * 100 + species_idx)
                submission_rows.append([row_id, prob])

        df_sub = pd.DataFrame(submission_rows, columns=["Id", "Probability"])
        df_sub.to_csv(CFG.submission_path, index=False)
        logger.info(f"Submission saved to {CFG.submission_path}")

    else:
        logger.info(
            f"Validation metric {final_val_metric} did not beat threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
