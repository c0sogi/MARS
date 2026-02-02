import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import library modules
from library.config import Config
from library.dataset import process_and_cache_data, WhaleDataset, get_augmentations
from library.models import get_model
from library.trainer import run_training
from library.utils import seed_everything, calculate_roc_auc, load_checkpoint


def predict(model, loader, device):
    """
    Runs inference on a DataLoader and returns probabilities.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(probs)
    return np.array(preds).flatten()


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Load Data
    print("Loading Metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)  # Hold-out Validation
    test_df = pd.read_csv(Config.TEST_CSV)

    print("Processing/Loading Datasets...")
    # Load all data tensors
    # Note: We use the 'train' prefix for the main training set which we will CV split
    full_train_data, full_train_targets = process_and_cache_data(
        train_df, "train", load_cached_data=True
    )
    holdout_data, holdout_targets = process_and_cache_data(
        val_df, "val", load_cached_data=True
    )
    test_data, test_clips = process_and_cache_data(
        test_df, "test", load_cached_data=True
    )

    # 3. Initialize Storage for Stacking
    # OOF predictions for the training set
    oof_preds = {
        "effnet": np.zeros(len(full_train_data)),
        "resnet": np.zeros(len(full_train_data)),
    }

    # Bagged predictions for Holdout and Test sets (summing folds, will divide later)
    holdout_preds_accum = {
        "effnet": np.zeros(len(holdout_data)),
        "resnet": np.zeros(len(holdout_data)),
    }
    test_preds_accum = {
        "effnet": np.zeros(len(test_data)),
        "resnet": np.zeros(len(test_data)),
    }

    # 4. Stratified K-Fold CV
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We need numpy array of targets for stratification
    train_targets_np = full_train_targets.numpy()

    # Create Holdout and Test Loaders once (reused across folds)
    holdout_dataset = WhaleDataset(holdout_data, holdout_targets, mode="val")
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_dataset = WhaleDataset(test_data, test_clips, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"\nStarting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_train_data, train_targets_np)
    ):
        print(f"\n=== Fold {fold} ===")

        # --- Prepare Fold Data ---
        fold_train_data = full_train_data[train_idx]
        fold_train_targets = full_train_targets[train_idx]
        fold_val_data = full_train_data[val_idx]
        fold_val_targets = full_train_targets[val_idx]

        # Weighted Sampler for Training
        class_counts = torch.bincount(fold_train_targets)
        class_weights = 1.0 / (class_counts.float() + 1e-6)  # Avoid div by zero
        sample_weights = class_weights[fold_train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        # Datasets & Loaders
        train_dataset = WhaleDataset(
            fold_train_data,
            fold_train_targets,
            mode="train",
            transform=get_augmentations(),
        )
        val_dataset = WhaleDataset(fold_val_data, fold_val_targets, mode="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # --- Train & Predict for Each Model ---
        models_to_train = [
            ("effnet", Config.MODEL_NAMES[0]),
            ("resnet", Config.MODEL_NAMES[1]),
        ]

        for model_key, model_name in models_to_train:
            print(f"Training {model_key} ({model_name})...")

            # Init Model
            model = get_model(model_name, pretrained=True).to(device)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )

            # Train
            run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                fold_idx=fold,
                model_name_suffix=model_key,
            )

            # Load Best Checkpoint
            checkpoint_path = os.path.join(
                Config.WORKING_DIR, f"{model_key}_fold_{fold}.pth"
            )
            load_checkpoint(model, checkpoint_path, device=device)

            # Predict OOF (Fold Validation)
            oof_probs = predict(model, val_loader, device)
            oof_preds[model_key][val_idx] = oof_probs

            # Predict Holdout (Bagging)
            holdout_probs = predict(model, holdout_loader, device)
            holdout_preds_accum[model_key] += holdout_probs

            # Predict Test (Bagging)
            test_probs = predict(model, test_loader, device)
            test_preds_accum[model_key] += test_probs

            # Free memory
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

    # 5. Stacking (Level 1)
    print("\n=== Training Meta-Learner (Stacking) ===")

    # Prepare Level 1 Training Data (OOF)
    X_train_stack = np.column_stack([oof_preds["effnet"], oof_preds["resnet"]])
    y_train_stack = train_targets_np

    # Prepare Level 1 Validation Data (Averaged Holdout)
    X_val_stack = np.column_stack(
        [
            holdout_preds_accum["effnet"] / Config.NUM_FOLDS,
            holdout_preds_accum["resnet"] / Config.NUM_FOLDS,
        ]
    )
    y_val_stack = holdout_targets.numpy()

    # Train Logistic Regression
    meta_model = LogisticRegression(random_state=Config.SEED)
    meta_model.fit(X_train_stack, y_train_stack)

    # Validate
    y_val_pred = meta_model.predict_proba(X_val_stack)[:, 1]
    final_auc = calculate_roc_auc(y_val_stack, y_val_pred)

    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error
    errors = np.abs(y_val_stack - y_val_pred)

    # Calculate features from Holdout Spectrograms (N, 1, F, T)
    # We compute stats over the last two dimensions (F, T)
    # Move to numpy for calculation
    holdout_data_np = holdout_data.numpy()

    feat_mean = np.mean(holdout_data_np, axis=(1, 2, 3))
    feat_std = np.std(holdout_data_np, axis=(1, 2, 3))
    feat_max = np.max(holdout_data_np, axis=(1, 2, 3))

    # Correlations
    corr_mean = np.corrcoef(errors, feat_mean)[0, 1]
    corr_std = np.corrcoef(errors, feat_std)[0, 1]
    corr_max = np.corrcoef(errors, feat_max)[0, 1]

    print("Correlation between Error Magnitude and Input Features:")
    print(f"Signal Mean: {corr_mean:.4f}")
    print(f"Signal Std : {corr_std:.4f}")
    print(f"Signal Max : {corr_max:.4f}")

    # 7. Submission
    THRESHOLD = 0.9959928858461402
    if final_auc > THRESHOLD:
        print("\nValidation metric passed threshold. Generating submission...")

        # Prepare Level 1 Test Data (Averaged Test)
        X_test_stack = np.column_stack(
            [
                test_preds_accum["effnet"] / Config.NUM_FOLDS,
                test_preds_accum["resnet"] / Config.NUM_FOLDS,
            ]
        )

        # Predict
        test_probs = meta_model.predict_proba(X_test_stack)[:, 1]

        # Create DataFrame
        submission_df = pd.DataFrame({"clip": test_clips, "probability": test_probs})

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_auc} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
