import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import cv2

# Library imports
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, calculate_class_weights
from library.dataset import AppleDataset, get_transforms
from library.model import AppleResNet34
from library.engine import train_one_epoch, validate


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Load Metadata
    # train_metadata.csv is the 80% split provided by metadata generation.
    # val_metadata.csv is the 20% hold-out split.
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_holdout = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Class Weights
    class_weights = calculate_class_weights(
        Config.TRAIN_METADATA_PATH, load_cached_data=True
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Stratified K-Fold on the training portion
    # We split the 'train_metadata' into internal train/val for the 5-fold process
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_models = []

    # Iterate Folds
    # We assume 'stratify_label' exists as per metadata generation script
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["stratify_label"])
    ):
        print(f"\n=== Fold {fold} ===")

        df_train_fold = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df_train_full.iloc[val_idx].reset_index(drop=True)

        # Initialize Model
        model = AppleResNet34(pretrained=Config.PRETRAINED).to(device)

        # ---------------------------
        # Phase 1: Coarse (256x256)
        # ---------------------------
        print("Phase 1: Coarse Training (256x256)")

        train_dataset_p1 = AppleDataset(
            df_train_fold, transform=get_transforms("train", Config.IMG_SIZE_PHASE_1)
        )
        val_dataset_p1 = AppleDataset(
            df_val_fold, transform=get_transforms("val", Config.IMG_SIZE_PHASE_1)
        )

        train_loader_p1 = DataLoader(
            train_dataset_p1,
            batch_size=Config.BATCH_SIZE_PHASE_1,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader_p1 = DataLoader(
            val_dataset_p1,
            batch_size=Config.BATCH_SIZE_PHASE_1,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        optimizer = optim.Adam(model.parameters(), lr=Config.LR_PHASE_1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS_PHASE_1
        )

        best_loss = float("inf")
        best_state_p1 = None

        for epoch in range(Config.EPOCHS_PHASE_1):
            train_loss = train_one_epoch(
                model, train_loader_p1, optimizer, criterion, device, epoch
            )
            val_loss, val_auc, _, _ = validate(model, val_loader_p1, criterion, device)
            scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state_p1 = model.state_dict()

        # Load best weights from Phase 1
        if best_state_p1 is not None:
            model.load_state_dict(best_state_p1)

        # ---------------------------
        # Phase 2: Fine (512x512)
        # ---------------------------
        print("Phase 2: Fine-Tuning (512x512)")

        train_dataset_p2 = AppleDataset(
            df_train_fold, transform=get_transforms("train", Config.IMG_SIZE_PHASE_2)
        )
        val_dataset_p2 = AppleDataset(
            df_val_fold, transform=get_transforms("val", Config.IMG_SIZE_PHASE_2)
        )

        train_loader_p2 = DataLoader(
            train_dataset_p2,
            batch_size=Config.BATCH_SIZE_PHASE_2,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader_p2 = DataLoader(
            val_dataset_p2,
            batch_size=Config.BATCH_SIZE_PHASE_2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Re-init optimizer with lower LR
        optimizer = optim.Adam(model.parameters(), lr=Config.LR_PHASE_2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.EPOCHS_PHASE_2
        )

        best_auc = 0.0
        best_model_path = os.path.join(Config.WORKING_DIR, f"resnet34_fold_{fold}.pth")

        for epoch in range(Config.EPOCHS_PHASE_2):
            train_loss = train_one_epoch(
                model, train_loader_p2, optimizer, criterion, device, epoch
            )
            val_loss, val_auc, _, _ = validate(model, val_loader_p2, criterion, device)
            scheduler.step()

            # Save best model based on AUC
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        fold_models.append(best_model_path)

        # Cleanup
        del (
            model,
            optimizer,
            scheduler,
            train_loader_p1,
            val_loader_p1,
            train_loader_p2,
            val_loader_p2,
        )
        torch.cuda.empty_cache()

    # ---------------------------
    # Hold-out Validation
    # ---------------------------
    print("\n=== Hold-out Validation ===")

    holdout_dataset = AppleDataset(
        df_holdout, transform=get_transforms("val", Config.INFERENCE_IMG_SIZE)
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=Config.BATCH_SIZE_PHASE_2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    avg_preds = np.zeros((len(df_holdout), Config.NUM_CLASSES))

    # Ensemble Inference
    for model_path in fold_models:
        model = AppleResNet34(pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in holdout_loader:
                images = batch["image"].to(device)

                # Forward
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                # TTA (Horizontal Flip)
                if Config.USE_TTA:
                    images_flip = torch.flip(images, dims=[3])
                    outputs_flip = model(images_flip)
                    probs_flip = torch.softmax(outputs_flip, dim=1)
                    probs = (probs + probs_flip) / 2.0

                fold_preds.append(probs.cpu().numpy())

        avg_preds += np.concatenate(fold_preds, axis=0)
        del model
        torch.cuda.empty_cache()

    avg_preds /= Config.N_FOLDS

    # Calculate Metric
    holdout_targets = df_holdout[Config.TARGET_COLS].values
    final_metric = calculate_roc_auc(holdout_targets, avg_preds)
    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------
    # Failure Analysis
    # ---------------------------
    print("\n=== Failure Analysis ===")

    # Calculate per-sample error (Cross Entropy)
    t_targets = torch.tensor(holdout_targets).to(device)
    t_preds = torch.tensor(avg_preds).to(device)
    t_preds = torch.clamp(t_preds, 1e-7, 1 - 1e-7)
    per_sample_loss = -torch.sum(t_targets * torch.log(t_preds), dim=1).cpu().numpy()

    # Extract meta-features
    widths, heights, intensities = [], [], []
    for idx, row in df_holdout.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(path)
        if img is not None:
            h, w, c = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            widths.append(w)
            heights.append(h)
            intensities.append(img_rgb.mean())
        else:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    # Correlations
    if np.std(widths) > 0:
        corr_w, _ = pearsonr(per_sample_loss, widths)
    else:
        corr_w = 0.0

    if np.std(heights) > 0:
        corr_h, _ = pearsonr(per_sample_loss, heights)
    else:
        corr_h = 0.0

    if np.std(intensities) > 0:
        corr_i, _ = pearsonr(per_sample_loss, intensities)
    else:
        corr_i = 0.0

    print(f"Correlation Error vs Width: {corr_w:.4f}")
    print(f"Correlation Error vs Height: {corr_h:.4f}")
    print(f"Correlation Error vs Intensity: {corr_i:.4f}")

    # ---------------------------
    # Submission
    # ---------------------------
    threshold = 0.9871488489626378
    if final_metric > threshold:
        print("\nMetric condition met. Generating submission...")

        test_dataset = AppleDataset(
            df_test,
            transform=get_transforms("test", Config.INFERENCE_IMG_SIZE),
            output_extra=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE_PHASE_2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        avg_test_preds = np.zeros((len(df_test), Config.NUM_CLASSES))
        image_ids = []

        for i, model_path in enumerate(fold_models):
            model = AppleResNet34(pretrained=False).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_test_preds = []
            current_ids = []

            with torch.no_grad():
                for batch in test_loader:
                    images = batch["image"].to(device)

                    if i == 0 and "image_id" in batch:
                        current_ids.extend(batch["image_id"])

                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)

                    if Config.USE_TTA:
                        images_flip = torch.flip(images, dims=[3])
                        outputs_flip = model(images_flip)
                        probs_flip = torch.softmax(outputs_flip, dim=1)
                        probs = (probs + probs_flip) / 2.0

                    fold_test_preds.append(probs.cpu().numpy())

            avg_test_preds += np.concatenate(fold_test_preds, axis=0)
            if i == 0:
                image_ids = current_ids

            del model
            torch.cuda.empty_cache()

        avg_test_preds /= Config.N_FOLDS

        # Create DataFrame
        sub_df = pd.DataFrame(avg_test_preds, columns=Config.TARGET_COLS)
        sub_df.insert(0, "image_id", image_ids)

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"Metric {final_metric} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
