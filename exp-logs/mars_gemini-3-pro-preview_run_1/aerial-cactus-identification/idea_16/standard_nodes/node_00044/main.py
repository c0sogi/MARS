import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    MetricMonitor,
    get_file_sizes,
    normalize_file_sizes,
)
from library.data_loader import load_and_cache_images, CactusDataset
from library.models import CactusRepVGG_MTL, CactusResNet_MTL, CactusNeXt_MTL
from library.train_engine import train_one_epoch, validate, SWAHandler
from library.inference_engine import reparameterize_model, predict_with_tta
from library.stacking import train_meta_learner, generate_final_predictions


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading Metadata and Images...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Load Images
    imgs_train = load_and_cache_images(df_train, "train_imgs_clean")
    imgs_val = load_and_cache_images(df_val, "val_imgs_clean")
    imgs_test = load_and_cache_images(df_test, "test_imgs_clean")

    # Load File Sizes
    sizes_train = get_file_sizes(df_train, cache_name="train_fsizes_clean")
    sizes_val = get_file_sizes(df_val, cache_name="val_fsizes_clean")
    sizes_test = get_file_sizes(df_test, cache_name="test_fsizes_clean")

    # Normalize File Sizes
    # This fits stats on train and applies to val/test
    fs_data = normalize_file_sizes(sizes_train, sizes_val, sizes_test)

    # Targets
    y_train = df_train["has_cactus"].values.astype(np.float32)
    y_val = df_val["has_cactus"].values.astype(np.float32)

    # Debug Subset
    if Config.DEBUG:
        print(f"Debug Mode: Slicing data to {Config.DEBUG_SUBSET_SIZE} samples")
        limit = Config.DEBUG_SUBSET_SIZE
        imgs_train = imgs_train[:limit]
        y_train = y_train[:limit]
        fs_data["train_film"] = fs_data["train_film"][:limit]
        fs_data["train_aux"] = fs_data["train_aux"][:limit]

        imgs_val = imgs_val[:limit]
        y_val = y_val[:limit]
        fs_data["val_film"] = fs_data["val_film"][:limit]
        fs_data["val_aux"] = fs_data["val_aux"][:limit]

        imgs_test = imgs_test[:limit]
        fs_data["test_film"] = fs_data["test_film"][:limit]
        df_test = df_test.iloc[:limit]

    # Transforms
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    eval_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # 3. Initialization for Stacking
    # OOF Predictions on Train Set: {Model: (N_train,)}
    oof_preds_train = {m: np.zeros(len(y_train)) for m in Config.MODELS_TO_RUN}

    # Predictions on Hold-out Val Set: {Model: List of (N_val,) per fold}
    val_preds_accum = {m: [] for m in Config.MODELS_TO_RUN}

    # Predictions on Test Set: {Model: List of (N_test,) per fold}
    test_preds_accum = {m: [] for m in Config.MODELS_TO_RUN}

    model_factory = {
        "RepVGG": CactusRepVGG_MTL,
        "ResNet": CactusResNet_MTL,
        "NeXt": CactusNeXt_MTL,
    }

    # 4. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    print(f"Starting {Config.N_FOLDS}-Fold CV on Training Set...")

    for fold, (train_idx, cv_val_idx) in enumerate(skf.split(imgs_train, y_train)):
        print(f"\n=== Fold {fold+1}/{Config.N_FOLDS} ===")

        # Create DataLoaders for this fold
        ds_fold_train = CactusDataset(
            imgs_train[train_idx],
            y_train[train_idx],
            fs_data["train_film"][train_idx],
            fs_data["train_aux"][train_idx],
            transform=train_transform,
        )
        ds_fold_val = CactusDataset(
            imgs_train[cv_val_idx],
            y_train[cv_val_idx],
            fs_data["train_film"][cv_val_idx],
            fs_data["train_aux"][cv_val_idx],
            transform=eval_transform,
        )

        dl_fold_train = torch.utils.data.DataLoader(
            ds_fold_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        dl_fold_val = torch.utils.data.DataLoader(
            ds_fold_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Fixed Validation Loader (Hold-out)
        ds_fixed_val = CactusDataset(
            imgs_val,
            y_val,
            fs_data["val_film"],
            fs_data["val_aux"],
            transform=eval_transform,
        )
        dl_fixed_val = torch.utils.data.DataLoader(
            ds_fixed_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Test Loader
        ds_test = CactusDataset(
            imgs_test, film_feats=fs_data["test_film"], transform=eval_transform
        )
        dl_test = torch.utils.data.DataLoader(
            ds_test,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Train each architecture
        for model_name in Config.MODELS_TO_RUN:
            print(f"  Training {model_name}...")

            # Initialize
            model = model_factory[model_name](num_classes=1).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )
            swa_handler = SWAHandler(model, device) if Config.USE_SWA else None

            best_auc = 0
            best_model_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{model_name}_fold{fold}_best.pth"
            )

            # Training Epochs
            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    model, dl_fold_train, optimizer, device, epoch
                )

                if swa_handler:
                    swa_handler.update(model, epoch)

                # Check performance on fold validation set
                auc, val_loss = validate(model, dl_fold_val, device)
                scheduler.step()

                if auc > best_auc:
                    best_auc = auc
                    torch.save(model.state_dict(), best_model_path)

            # Finalize Model (SWA or Best)
            final_model = None
            if swa_handler and (Config.EPOCHS > Config.SWA_START_EPOCH):
                # print("    Finalizing SWA...")
                final_model = swa_handler.finalize(dl_fold_train)
            else:
                # print("    Loading Best Model...")
                model.load_state_dict(torch.load(best_model_path))
                final_model = model

            # Prepare for Inference
            final_model = reparameterize_model(final_model)

            # 1. OOF Prediction (on fold validation part of train set)
            oof_p = predict_with_tta(final_model, dl_fold_val, device)
            oof_preds_train[model_name][cv_val_idx] = oof_p

            # 2. Hold-out Validation Prediction
            val_p = predict_with_tta(final_model, dl_fixed_val, device)
            val_preds_accum[model_name].append(val_p)

            # 3. Test Prediction
            test_p = predict_with_tta(final_model, dl_test, device)
            test_preds_accum[model_name].append(test_p)

            # Cleanup
            del model, final_model, optimizer, scheduler, swa_handler
            torch.cuda.empty_cache()

    # 5. Stacking & Meta-Learning
    print("\n=== Stacking & Meta-Learning ===")

    # Train Meta-Learner on Train OOF
    meta_model = train_meta_learner(
        oof_preds_train, y_train, meta_features=fs_data["train_film"]
    )

    # Prepare Averaged Predictions for Validation
    val_preds_avg = {
        m: np.mean(np.stack(val_preds_accum[m]), axis=0) for m in Config.MODELS_TO_RUN
    }

    # Predict on Hold-out Validation
    from library.stacking import prepare_meta_features

    X_val_meta, _ = prepare_meta_features(val_preds_avg, fs_data["val_film"])
    val_final_probs = meta_model.predict_proba(X_val_meta)[:, 1]

    # Calculate Final Metric
    final_metric = roc_auc_score(y_val, val_final_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_val - val_final_probs)
    # Correlate errors with normalized file size
    corr, p_val = pearsonr(errors, fs_data["val_film"])
    print(f"Correlation between Error and File Size: {corr:.4f} (p={p_val:.4f})")

    # 7. Submission
    print("\n=== Generating Submission ===")
    # Prepare Averaged Predictions for Test
    test_preds_avg = {
        m: np.mean(np.stack(test_preds_accum[m]), axis=0) for m in Config.MODELS_TO_RUN
    }

    generate_final_predictions(
        meta_model,
        test_preds_avg,
        df_test["id"].values,
        meta_features=fs_data["test_film"],
    )


if __name__ == "__main__":
    main()
