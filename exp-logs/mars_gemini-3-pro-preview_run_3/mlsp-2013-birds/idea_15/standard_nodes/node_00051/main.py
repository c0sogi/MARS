import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from provided library
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, ModelEMA
from library.dataset import preprocess_and_cache_images, BirdDataset, get_transforms
from library.models import BirdModel
from library.training import train_fold

# Try importing IterativeStratification
try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_SKMULTILEARN = True
except ImportError:
    HAS_SKMULTILEARN = False
    from sklearn.model_selection import KFold


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # Adjust Config for runtime constraints
    # The dataset is small (206 samples), so 30 epochs is sufficient and fast.
    Config.EPOCHS = 30

    device = Config.DEVICE

    # 2. Data Preparation
    # Load Cache
    images_cache = preprocess_and_cache_images(load_cached_data=True)

    # Load Metadata
    train_df_orig = pd.read_csv(Config.TRAIN_CSV)
    val_df_holdout = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Prepare Labels for Stratification
    num_classes = Config.NUM_CLASSES
    X = train_df_orig["rec_id"].values.reshape(-1, 1)

    # Parse labels into binary matrix for stratification
    y_train = np.zeros((len(train_df_orig), num_classes), dtype=int)
    for idx, row in train_df_orig.iterrows():
        lbl_str = str(row["labels"])
        if lbl_str != "?" and lbl_str.strip():
            try:
                indices = [int(x) for x in lbl_str.split()]
                for cls_idx in indices:
                    if 0 <= cls_idx < num_classes:
                        y_train[idx, cls_idx] = 1
            except:
                pass

    # 3. Define Folds
    folds = []
    if HAS_SKMULTILEARN:
        try:
            stratifier = IterativeStratification(n_splits=Config.N_FOLDS, order=1)
            # IterativeStratification returns train_indices, test_indices
            for train_idx, val_idx in stratifier.split(X, y_train):
                folds.append((train_idx, val_idx))
        except Exception as e:
            print(f"IterativeStratification failed: {e}. Falling back to Random Split.")

    if not folds:
        # Fallback to KFold
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
        for train_idx, val_idx in kf.split(X):
            folds.append((train_idx, val_idx))

    # 4. Training Loop
    # Store model paths for ensemble
    model_checkpoints = []

    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        print(f"\n=== Fold {fold_idx} ===")

        # Create DataFrames for this fold
        fold_train_df = train_df_orig.iloc[train_indices].reset_index(drop=True)
        fold_val_df = train_df_orig.iloc[val_indices].reset_index(drop=True)

        # Create Datasets & Loaders
        train_ds = BirdDataset(
            fold_train_df, images_cache, transforms=get_transforms("train")
        )
        val_ds = BirdDataset(
            fold_val_df, images_cache, transforms=get_transforms("val")
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Train each backbone
        for backbone in Config.BACKBONES:
            print(f"Training {backbone}...")

            # Initialize Model
            model = BirdModel(
                backbone_name=backbone, num_classes=Config.NUM_CLASSES, pretrained=True
            )
            model.to(device)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
            )

            # Train
            model, best_auc = train_fold(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                fold_idx,
                num_epochs=Config.EPOCHS,
                patience=Config.PATIENCE,
            )

            # Store checkpoint path
            ckpt_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{backbone}_fold_{fold_idx}_best.pth"
            )
            model_checkpoints.append({"backbone": backbone, "path": ckpt_path})

            # Clear memory
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

    # 5. Validation on Hold-out Set (Ensemble)
    print("\n=== Final Validation on Hold-out Set ===")

    # Load Hold-out Data
    holdout_ds = BirdDataset(
        val_df_holdout, images_cache, transforms=get_transforms("val")
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Prepare Ground Truth
    y_true = np.zeros((len(val_df_holdout), num_classes), dtype=int)
    for idx, row in val_df_holdout.iterrows():
        lbl_str = str(row["labels"])
        if lbl_str != "?" and lbl_str.strip():
            try:
                indices = [int(x) for x in lbl_str.split()]
                for cls_idx in indices:
                    if 0 <= cls_idx < num_classes:
                        y_true[idx, cls_idx] = 1
            except:
                pass

    # Collect predictions from all models
    ensemble_preds = []

    for ckpt_info in model_checkpoints:
        backbone = ckpt_info["backbone"]
        path = ckpt_info["path"]

        # Load Model
        model = BirdModel(
            backbone_name=backbone, num_classes=Config.NUM_CLASSES, pretrained=False
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        preds = []
        with torch.no_grad():
            for images, _, _ in holdout_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                preds.append(probs.cpu().numpy())

        if preds:
            ensemble_preds.append(np.concatenate(preds))

        del model
        torch.cuda.empty_cache()

    # Average Predictions
    if ensemble_preds:
        avg_preds = np.mean(ensemble_preds, axis=0)
        final_auc = calculate_roc_auc(y_true, avg_preds)
    else:
        final_auc = 0.5
        avg_preds = np.zeros_like(y_true, dtype=float)

    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude (Mean Absolute Error per sample, averaged over classes)
    errors = np.mean(np.abs(y_true - avg_preds), axis=1)

    # Calculate Input Features (Image Mean intensity)
    feature_values = []
    for idx, row in val_df_holdout.iterrows():
        rec_id = int(row["rec_id"])
        img = images_cache[rec_id]  # (H, W)
        feature_values.append(np.mean(img))

    feature_values = np.array(feature_values)

    # Correlation
    if len(errors) > 1:
        correlation = np.corrcoef(errors, feature_values)[0, 1]
    else:
        correlation = 0.0

    print(f"Correlation between Error Magnitude and Image Intensity: {correlation}")

    # 7. Submission
    threshold = 0.9479806884980326
    if final_auc > threshold:
        print("\n=== Generating Submission ===")

        test_ds = BirdDataset(
            test_df, images_cache, transforms=get_transforms("test"), is_test=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_ensemble_preds = []
        rec_ids = []

        # Get Rec IDs
        for _, _, ids in test_loader:
            rec_ids.append(ids.numpy())
        if rec_ids:
            rec_ids = np.concatenate(rec_ids)

        # Predict with all models
        for ckpt_info in model_checkpoints:
            backbone = ckpt_info["backbone"]
            path = ckpt_info["path"]

            model = BirdModel(
                backbone_name=backbone, num_classes=Config.NUM_CLASSES, pretrained=False
            )
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()

            preds = []
            with torch.no_grad():
                for images, _, _ in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                    preds.append(probs.cpu().numpy())

            if preds:
                test_ensemble_preds.append(np.concatenate(preds))

            del model
            torch.cuda.empty_cache()

        if test_ensemble_preds:
            avg_test_preds = np.mean(test_ensemble_preds, axis=0)

            # Format Submission
            # Id = rec_id * 100 + species_id
            submission_rows = []
            for i in range(len(rec_ids)):
                r_id = rec_ids[i]
                probs = avg_test_preds[i]
                for species_id in range(num_classes):
                    row_id = r_id * 100 + species_id
                    submission_rows.append(
                        {"Id": row_id, "Probability": probs[species_id]}
                    )

            sub_df = pd.DataFrame(submission_rows)
            sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation Metric {final_auc} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
