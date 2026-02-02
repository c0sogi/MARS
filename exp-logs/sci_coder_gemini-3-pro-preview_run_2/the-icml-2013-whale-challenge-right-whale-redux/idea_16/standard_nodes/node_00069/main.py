import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_auc, EarlyStopping
from library.dataset import WhaleDataset, get_train_transforms
from library.models import WhaleModel
from library.engine import train_one_epoch, validate
from library.ensemble import MetaLearner, aggregate_predictions, save_submission


def run():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    # We need two versions of the training set: one with augmentation for training,
    # and one without for generating OOF predictions.
    train_ds_aug = WhaleDataset(mode="train", transform=get_train_transforms())
    train_ds_clean = WhaleDataset(mode="train", transform=None)

    # Hold-out Validation Set (never seen during training)
    val_ds = WhaleDataset(mode="val", transform=None)

    # Test Set
    test_ds = WhaleDataset(mode="test", transform=None)

    # Pre-load targets for stratified splitting
    train_targets = train_ds_aug.targets.astype(int)

    # 3. Cross-Validation Setup
    n_folds = Config.N_FOLDS
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    # Storage for predictions
    # OOF: (N_train, N_models) - Used to train Meta-Learner
    oof_preds = {
        model_name: np.zeros(len(train_ds_aug)) for model_name in Config.MODEL_NAMES
    }

    # Hold-out Val: List of arrays (one per fold) -> to be averaged (Bagging)
    val_preds_fold = {model_name: [] for model_name in Config.MODEL_NAMES}

    # Test: List of arrays (one per fold) -> to be averaged (Bagging)
    test_preds_fold = {model_name: [] for model_name in Config.MODEL_NAMES}

    # 4. Training Loop
    # Limit epochs and patience for a fast baseline execution
    MAX_EPOCHS = 10
    PATIENCE = 3

    print(f"Starting {n_folds}-Fold Cross-Validation...")

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(np.zeros(len(train_targets)), train_targets)
    ):
        print(f"\n{'='*20} Fold {fold+1}/{n_folds} {'='*20}")

        # --- Prepare DataLoaders for this fold ---

        # Train Subset (Augmented)
        train_sub = Subset(train_ds_aug, train_idx)

        # Weighted Random Sampler for Class Imbalance
        fold_targets = train_targets[train_idx]
        class_counts = np.bincount(fold_targets)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[fold_targets]

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_sub,
            batch_size=Config.BATCH_SIZE,
            sampler=sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # Validation/OOF Subset (Clean)
        valid_sub = Subset(train_ds_clean, valid_idx)
        valid_loader = DataLoader(
            valid_sub,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Hold-out Val Loader (Full val set)
        holdout_val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Test Loader
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # --- Train each model architecture ---
        for model_name in Config.MODEL_NAMES:
            print(f"\n--- Training {model_name} (Fold {fold+1}) ---")

            # Initialize Model
            model = WhaleModel(model_name=model_name, pretrained=True)
            model.to(device)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=MAX_EPOCHS, eta_min=Config.MIN_LR
            )
            criterion = nn.BCEWithLogitsLoss()

            # Early Stopping
            save_path = os.path.join(Config.OUTPUT_DIR, f"{model_name}_fold{fold}.pth")
            es = EarlyStopping(
                patience=PATIENCE, mode=Config.EARLY_STOPPING_MODE, save_path=save_path
            )

            # Epoch Loop
            for epoch in range(MAX_EPOCHS):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, device, criterion
                )
                val_loss, val_auc = validate(model, valid_loader, device, criterion)
                scheduler.step()

                print(
                    f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
                )

                es(val_auc, model)
                if es.early_stop:
                    print("Early stopping triggered")
                    break

            # Load Best Model for Inference
            model.load_state_dict(torch.load(save_path))
            model.eval()

            # Helper for inference
            def get_preds(loader, is_test=False):
                preds = []
                with torch.no_grad():
                    for batch in loader:
                        inputs = batch[0].to(device)
                        outputs = model(inputs)
                        probs = torch.sigmoid(outputs).cpu().numpy().ravel()
                        preds.append(probs)
                return np.concatenate(preds)

            # 1. Generate OOF Predictions (for Meta-Learner training)
            oof_probs = get_preds(valid_loader)
            oof_preds[model_name][valid_idx] = oof_probs

            # 2. Predict on Hold-out Validation (for Meta-Learner evaluation)
            val_probs = get_preds(holdout_val_loader)
            val_preds_fold[model_name].append(val_probs)

            # 3. Predict on Test (for Submission)
            test_probs = get_preds(test_loader, is_test=True)
            test_preds_fold[model_name].append(test_probs)

            # Cleanup to save memory
            del model, optimizer, scheduler, es
            torch.cuda.empty_cache()
            gc.collect()

    # 5. Ensemble (Stacking)
    print("\n" + "=" * 20 + " Ensembling " + "=" * 20)

    # Train Meta-Learner on OOF predictions
    meta_learner = MetaLearner()
    meta_auc = meta_learner.fit(oof_preds, train_targets)

    # 6. Final Validation & Metric
    # Aggregate hold-out predictions (Bagging across folds)
    avg_val_preds = {}
    for model_name in Config.MODEL_NAMES:
        avg_val_preds[model_name] = aggregate_predictions(val_preds_fold[model_name])

    # Predict using Meta-Learner
    final_val_probs = meta_learner.predict(avg_val_preds)

    # Calculate Final Metric on Hold-out Set
    val_targets = val_ds.targets
    final_auc = calculate_auc(val_targets, final_val_probs)

    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\n" + "=" * 20 + " Failure Analysis " + "=" * 20)
    # Calculate error magnitude
    errors = np.abs(val_targets - final_val_probs)

    # Compute input features from validation data for correlation analysis
    # We use simple statistics of the spectrograms: Mean, Std, Max
    print("Computing input features for failure analysis...")
    val_data = val_ds.data  # Access numpy array directly

    # Flatten spatial dimensions: (N, 1, F, T) -> (N, F*T)
    flat_data = val_data.reshape(len(val_data), -1)

    feat_mean = np.mean(flat_data, axis=1)
    feat_std = np.std(flat_data, axis=1)
    feat_max = np.max(flat_data, axis=1)

    # Calculate Correlations
    corr_mean, _ = pearsonr(errors, feat_mean)
    corr_std, _ = pearsonr(errors, feat_std)
    corr_max, _ = pearsonr(errors, feat_max)

    print(f"Correlation (Error vs Input Mean): {corr_mean:.10f}")
    print(f"Correlation (Error vs Input Std): {corr_std:.10f}")
    print(f"Correlation (Error vs Input Max): {corr_max:.10f}")

    # 8. Submission
    THRESHOLD = 0.9961020185223004
    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Aggregate Test Preds
        avg_test_preds = {}
        for model_name in Config.MODEL_NAMES:
            avg_test_preds[model_name] = aggregate_predictions(
                test_preds_fold[model_name]
            )

        # Meta-Learner Prediction
        final_test_probs = meta_learner.predict(avg_test_preds)

        # Get Clip Names
        test_clips = test_ds.targets  # For test set, targets are clip names

        save_submission(test_clips, final_test_probs)
    else:
        print(
            f"\nMetric ({final_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
