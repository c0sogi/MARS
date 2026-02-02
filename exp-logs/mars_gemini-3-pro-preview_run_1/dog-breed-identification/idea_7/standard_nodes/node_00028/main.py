import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calc_log_loss, MixupCutmix
from library.dataset import get_datasets, get_transforms, DogDataset
from library.model import get_model, set_backbone_trainable
from library.engine import (
    train_one_epoch,
    validate,
    predict,
    update_bn_statistics,
    save_submission,
)


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Loading datasets...")
    # Load all data (Train, Val Hold-out, Test)
    # train_data and val_data are tuples (images, labels)
    (
        (train_images, train_labels),
        (val_holdout_images, val_holdout_labels),
        (test_images, _),
        label_map,
    ) = get_datasets(load_cached_data=True)

    # 3. Training Loop (Single Strong Model Ensemble)
    models_preds_val = []  # To store predictions on hold-out val set from each model
    models_preds_test = []  # To store predictions on test set from each model

    # We will train 1 architecture * 5 folds = 5 models (Cite solution_lesson_node_00021)
    model_keys = list(Config.MODEL_CONFIGS.keys())

    # Prepare Hold-out Validation Loader (used for final ensemble eval, not training CV)

    for model_key in model_keys:
        print(f"\n=== Starting Training for Architecture: {model_key} ===")
        model_cfg = Config.MODEL_CONFIGS[model_key]
        input_size = model_cfg["input_size"]

        # Prepare K-Fold
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        # Transforms
        train_transform = get_transforms(mode="train", input_size=input_size)
        val_transform = get_transforms(mode="val", input_size=input_size)

        # Hold-out Validation Dataset/Loader for this architecture
        val_holdout_ds = DogDataset(
            val_holdout_images, val_holdout_labels, transforms=val_transform
        )
        val_holdout_loader = DataLoader(
            val_holdout_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Test Dataset/Loader for this architecture
        test_ds = DogDataset(test_images, None, transforms=val_transform)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_images, train_labels)
        ):
            print(f"--- Fold {fold + 1}/{Config.N_FOLDS} ---")

            # Split Data
            X_train, y_train = train_images[train_idx], train_labels[train_idx]
            X_val, y_val = train_images[val_idx], train_labels[val_idx]

            # Datasets
            train_ds = DogDataset(X_train, y_train, transforms=train_transform)
            val_ds = DogDataset(X_val, y_val, transforms=val_transform)

            # Loaders
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

            # Initialize Model
            model = get_model(model_cfg)
            model.to(device)

            # Initialize Mixup
            mixup_fn = MixupCutmix(
                mixup_alpha=Config.TRAINING_PHASES["phase_2"]["mixup_alpha"],
                cutmix_alpha=Config.TRAINING_PHASES["phase_2"]["cutmix_alpha"],
                mix_prob=Config.TRAINING_PHASES["phase_2"]["mix_prob"],
                num_classes=model_cfg["num_classes"],
            )

            # --- Phase 1: Head Adaptation ---
            print("Phase 1: Head Adaptation")
            cfg_p1 = Config.TRAINING_PHASES["phase_1"]
            set_backbone_trainable(model, trainable=False)
            optimizer = AdamW(model.parameters(), lr=cfg_p1["lr"])

            for epoch in range(cfg_p1["epochs"]):
                train_one_epoch(
                    model, optimizer, train_loader, device, epoch, mixup_fn=None
                )

            # --- Phase 2: Regularized Fine-Tuning ---
            print("Phase 2: Regularized Fine-Tuning")
            cfg_p2 = Config.TRAINING_PHASES["phase_2"]
            set_backbone_trainable(model, trainable=True)

            # Discriminative LRs (Backbone vs Head)
            param_groups = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "head" not in n and "fc" not in n and "classifier" not in n
                    ],
                    "lr": cfg_p2["lr_backbone"],
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if "head" in n or "fc" in n or "classifier" in n
                    ],
                    "lr": cfg_p2["lr_head"],
                },
            ]
            optimizer = AdamW(param_groups)
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg_p2["epochs"])

            for epoch in range(cfg_p2["epochs"]):
                train_one_epoch(
                    model, optimizer, train_loader, device, epoch, mixup_fn=mixup_fn
                )
                scheduler.step()

            # --- Phase 3: Regularization Cooldown ---
            print("Phase 3: Regularization Cooldown")
            cfg_p3 = Config.TRAINING_PHASES["phase_3"]
            # Re-init optimizer with low LR
            optimizer = AdamW(model.parameters(), lr=cfg_p3["lr"])

            for epoch in range(cfg_p3["epochs"]):
                train_one_epoch(
                    model, optimizer, train_loader, device, epoch, mixup_fn=None
                )

            # --- Phase 4: SWA ---
            print("Phase 4: SWA")
            cfg_p4 = Config.TRAINING_PHASES["phase_4"]
            swa_model = AveragedModel(model)
            optimizer = AdamW(model.parameters(), lr=cfg_p4["lr"])

            for epoch in range(cfg_p4["epochs"]):
                train_one_epoch(
                    model, optimizer, train_loader, device, epoch, mixup_fn=None
                )
                swa_model.update_parameters(model)

            # Update BN statistics for SWA model
            update_bn_statistics(swa_model, train_loader, device)

            # Use SWA model for inference
            final_model = swa_model

            # Save Model (Optional but good practice)
            save_path = os.path.join(Config.OUTPUT_DIR, f"{model_key}_fold_{fold}.pth")
            torch.save(final_model.state_dict(), save_path)

            # --- Inference for Ensemble ---
            # 1. Predict on Hold-out Validation Set
            print("Generating validation predictions...")
            preds_val = predict(
                final_model, val_holdout_loader, device, use_tta=Config.USE_TTA
            )
            models_preds_val.append(preds_val)

            # 2. Predict on Test Set (Stored for potential submission)
            print("Generating test predictions...")
            preds_test = predict(
                final_model, test_loader, device, use_tta=Config.USE_TTA
            )
            models_preds_test.append(preds_test)

            # Clean up to save memory
            del model, swa_model, final_model, optimizer, scheduler
            torch.cuda.empty_cache()

    # 4. Validation & Ensemble Aggregation
    print("\n=== Ensemble Evaluation ===")
    # Average predictions across all models
    ensemble_preds_val = np.mean(models_preds_val, axis=0)

    # Calculate Metric
    # val_holdout_labels is (N,)
    final_metric = calc_log_loss(val_holdout_labels, ensemble_preds_val)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate per-sample log loss
    # Extract probability assigned to the true class
    # y_true indices
    y_true = val_holdout_labels
    # Get prob of true class: preds[i, y_true[i]]
    # Clip to avoid log(0)
    probs_true = ensemble_preds_val[np.arange(len(y_true)), y_true]
    probs_true = np.clip(probs_true, 1e-15, 1 - 1e-15)
    sample_losses = -np.log(probs_true)

    # Get Original Image Metadata for Validation Set
    # We read from the metadata file to get original dimensions, as cached images are already resized.
    val_df = pd.read_csv(Config.VAL_METADATA)
    val_widths = []
    val_heights = []
    val_ratios = []

    print("Analyzing original image dimensions for failure analysis...")
    for _, row in val_df.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            val_widths.append(w)
            val_heights.append(h)
            val_ratios.append(w / h if h > 0 else 0)
        else:
            val_widths.append(0)
            val_heights.append(0)
            val_ratios.append(0)

    val_widths = np.array(val_widths)
    val_heights = np.array(val_heights)
    val_ratios = np.array(val_ratios)

    # Calculate correlations
    corr_width = np.corrcoef(sample_losses, val_widths)[0, 1]
    corr_height = np.corrcoef(sample_losses, val_heights)[0, 1]
    corr_ratio = np.corrcoef(sample_losses, val_ratios)[0, 1]

    print(f"Correlation between Error and Image Width: {corr_width:.4f}")
    print(f"Correlation between Error and Image Height: {corr_height:.4f}")
    print(f"Correlation between Error and Aspect Ratio: {corr_ratio:.4f}")

    # 6. Submission
    THRESHOLD = 0.12970461086690332
    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        ensemble_preds_test = np.mean(models_preds_test, axis=0)
        save_submission(
            ensemble_preds_test, Config.TEST_METADATA, label_map, Config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nValidation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
