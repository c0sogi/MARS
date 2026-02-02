import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratification
from scipy.stats import pearsonr
import cv2

# Import from library
from library.config import Config
from library.utils import set_seed, calculate_pos_weights, compute_auc
from library.dataset import load_images, BirdDataset
from library.transforms import get_transforms, cyclic_time_roll
from library.models import BirdClassifier
from library.losses import WeightedDistillationLoss
from library.engine import Engine


def predict_with_tta(engine, loader, device):
    """
    Performs inference with Cyclic Test-Time Augmentation (TTA).
    Shifts: [0, 0.25, 0.5, 0.75]
    """
    engine.model.eval()
    shifts = [0.0, 0.25, 0.5, 0.75]
    accumulated_preds = None

    # We need to iterate through the loader for each shift
    # Since we can't easily modify the loader in-place efficiently for all batches without reloading,
    # we will iterate the loader once per shift.

    for shift in shifts:
        preds_list = []
        with torch.no_grad():
            for data in loader:
                images = data["image"].numpy()  # Convert to numpy for rolling

                # Apply cyclic roll
                # images is (B, C, H, W)
                # cyclic_time_roll expects (H, W, C) or (C, H, W)?
                # The utils function handles axis=1 for 3D or 2D.
                # Albumentations works on HWC. The loader returns CHW tensors.
                # Let's handle it carefully.

                # Convert back to HWC for the transform function if needed,
                # or just roll on axis 3 (Width) of the tensor (B, C, H, W).

                # Implementation of roll on tensor directly for speed
                if shift > 0:
                    width = images.shape[3]
                    shift_pixels = int(width * shift)
                    images = np.roll(images, shift_pixels, axis=3)

                images_tensor = torch.from_numpy(images).to(device)

                logits = engine.model(images_tensor)
                preds = torch.sigmoid(logits).cpu().numpy()
                preds_list.append(preds)

        current_shift_preds = np.concatenate(preds_list, axis=0)

        if accumulated_preds is None:
            accumulated_preds = current_shift_preds
        else:
            accumulated_preds += current_shift_preds

    return accumulated_preds / len(shifts)


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading Metadata...")
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Load Images (Cached)
    print("Loading Images...")
    train_images = load_images(df_train, "train", load_cached_data=True)
    val_images = load_images(df_val, "val", load_cached_data=True)
    test_images = load_images(df_test, "test", load_cached_data=True)

    # Calculate Positive Weights for Loss
    label_cols = [c for c in df_train.columns if c.startswith("species_")]
    y_train_all = df_train[label_cols].values
    pos_weights = calculate_pos_weights(
        y_train_all, factor=Config.POS_WEIGHT_FACTOR
    ).to(device)

    # 3. Generation 0: Train Anchors & Generate Soft Targets
    print("\n=== Generation 0: Training Anchors ===")

    # We need OOF predictions for the entire train set.
    # We use Iterative Stratified K-Fold.
    n_folds = 5
    k_fold = IterativeStratification(n_splits=n_folds, order=1)

    # Placeholder for OOF predictions
    # We have 2 anchors: ResNet18, EfficientNet_B0
    oof_preds_resnet = np.zeros((len(df_train), Config.NUM_CLASSES))
    oof_preds_effnet = np.zeros((len(df_train), Config.NUM_CLASSES))

    # Dummy X for splitting
    X_dummy = np.zeros((len(df_train), 1))

    # Training Parameters for Gen 0
    gen0_epochs = 10
    batch_size = Config.BATCH_SIZE

    fold = 0
    for train_idx, val_idx in k_fold.split(X_dummy, y_train_all):
        print(f"  Processing Fold {fold+1}/{n_folds}...")

        # Split Data
        fold_train_imgs = train_images[train_idx]
        fold_val_imgs = train_images[val_idx]
        fold_train_df = df_train.iloc[train_idx].reset_index(drop=True)
        fold_val_df = df_train.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_ds = BirdDataset(
            fold_train_imgs, fold_train_df, transforms=get_transforms("train")
        )
        val_ds = BirdDataset(
            fold_val_imgs, fold_val_df, transforms=get_transforms("val")
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # --- Train ResNet18 Anchor ---
        model_res = BirdClassifier("resnet18", pretrained=True).to(device)
        optimizer = torch.optim.AdamW(
            model_res.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        # Standard BCE for Gen 0
        loss_fn = WeightedDistillationLoss(
            pos_weight=pos_weights, distillation_lambda=0.0
        )

        engine_res = Engine(model_res, device, optimizer, loss_fn=loss_fn)
        engine_res.fit(
            train_loader,
            val_loader,
            epochs=gen0_epochs,
            patience=3,
            save_path=f"{Config.CACHE_DIR}/gen0_res_fold{fold}.pth",
        )

        # Predict OOF
        oof_preds_resnet[val_idx] = engine_res.predict(val_loader)

        # --- Train EfficientNet Anchor ---
        model_eff = BirdClassifier("efficientnet_b0", pretrained=True).to(device)
        optimizer = torch.optim.AdamW(
            model_eff.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        engine_eff = Engine(model_eff, device, optimizer, loss_fn=loss_fn)
        engine_eff.fit(
            train_loader,
            val_loader,
            epochs=gen0_epochs,
            patience=3,
            save_path=f"{Config.CACHE_DIR}/gen0_eff_fold{fold}.pth",
        )

        # Predict OOF
        oof_preds_effnet[val_idx] = engine_eff.predict(val_loader)

        fold += 1

    # Aggregate Soft Targets
    soft_targets = (oof_preds_resnet + oof_preds_effnet) / 2.0
    print("Generation 0 Complete. Soft Targets Generated.")

    # 4. Generation 1: Train Full Ensemble with Distillation
    print("\n=== Generation 1: Training Students (Born-Again) ===")

    # We train on the full training set (using the generated soft targets)
    # and validate on the hold-out validation set.

    gen1_epochs = 15

    # Create Dataset with Soft Labels
    train_ds_full = BirdDataset(
        train_images,
        df_train,
        transforms=get_transforms("train"),
        soft_labels=soft_targets,
    )
    val_ds_holdout = BirdDataset(val_images, df_val, transforms=get_transforms("val"))

    train_loader_full = DataLoader(
        train_ds_full,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader_holdout = DataLoader(
        val_ds_holdout, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Models to train
    model_names = ["resnet18", "efficientnet_b0", "densenet121"]
    trained_models = []

    for name in model_names:
        print(f"  Training {name}...")
        model = BirdClassifier(name, pretrained=True).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Weighted Distillation Loss
        loss_fn = WeightedDistillationLoss(
            pos_weight=pos_weights, distillation_lambda=Config.DISTILLATION_LAMBDA
        )

        engine = Engine(model, device, optimizer, loss_fn=loss_fn)
        engine.fit(
            train_loader_full,
            val_loader_holdout,
            epochs=gen1_epochs,
            patience=5,
            save_path=f"{Config.CACHE_DIR}/gen1_{name}.pth",
        )
        trained_models.append(engine)

    # 5. Validation & Inference
    print("\n=== Inference & Evaluation ===")

    # Predictions on Validation Set (Ensemble + TTA)
    val_preds_ensemble = np.zeros((len(df_val), Config.NUM_CLASSES))

    for engine in trained_models:
        preds = predict_with_tta(engine, val_loader_holdout, device)
        val_preds_ensemble += preds

    val_preds_ensemble /= len(trained_models)

    # Compute Final Metric
    y_val = df_val[label_cols].values
    final_auc = compute_auc(y_val, val_preds_ensemble)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate per-sample error (Mean Absolute Error across classes)
    per_sample_error = np.mean(np.abs(y_val - val_preds_ensemble), axis=1)

    # Compute metadata features for correlation
    pixel_means = []
    pixel_stds = []
    for img in val_images:
        # img is H,W,3. Convert to float for stats
        img_f = img.astype(float) / 255.0
        pixel_means.append(np.mean(img_f))
        pixel_stds.append(np.std(img_f))

    corr_mean, _ = pearsonr(per_sample_error, pixel_means)
    corr_std, _ = pearsonr(per_sample_error, pixel_stds)

    print(f"Correlation (Error vs Pixel Mean): {corr_mean:.4f}")
    print(f"Correlation (Error vs Pixel Std): {corr_std:.4f}")

    # 7. Submission
    threshold = 0.0
    if final_auc > threshold:
        print(f"Validation score {final_auc} > {threshold}. Generating submission...")

        test_ds = BirdDataset(
            test_images, df_test, transforms=get_transforms("val")
        )  # TTA handles transforms
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )

        test_preds_ensemble = np.zeros((len(df_test), Config.NUM_CLASSES))

        for engine in trained_models:
            preds = predict_with_tta(engine, test_loader, device)
            test_preds_ensemble += preds

        test_preds_ensemble /= len(trained_models)

        # Flatten for submission format
        # Format: Id,Probability
        # Id = rec_id * 100 + species_id
        submission_rows = []
        rec_ids = df_test["rec_id"].values

        for i, rec_id in enumerate(rec_ids):
            probs = test_preds_ensemble[i]
            for species_idx, prob in enumerate(probs):
                row_id = rec_id * 100 + species_idx
                submission_rows.append([row_id, prob])

        submission_df = pd.DataFrame(submission_rows, columns=["Id", "Probability"])

        # Sort by Id just in case
        submission_df = submission_df.sort_values("Id")

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"Validation score {final_auc} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
