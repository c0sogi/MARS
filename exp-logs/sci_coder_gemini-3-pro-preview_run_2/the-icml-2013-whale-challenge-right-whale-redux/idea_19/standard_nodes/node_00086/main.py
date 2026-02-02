import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import library modules
from library import config, utils, dataset, models, engine, stacking


def main():
    # 1. Configuration and Setup
    utils.seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override config for fast baseline execution while maintaining ensemble structure
    config.NUM_EPOCHS = (
        15  # Increased epochs for convergence (Cite solution_lesson_node_00026)
    )
    config.N_FOLDS = 5

    # 2. Data Loading
    # We load both train and val sets provided by metadata and combine them
    # to perform our own Stratified 5-Fold CV as per the Idea.
    print("Loading data...")
    train_X, train_y, _ = dataset.get_data(
        config.TRAIN_CSV, "train", load_cached_data=True
    )
    val_X, val_y, _ = dataset.get_data(config.VAL_CSV, "val", load_cached_data=True)

    # Combine for Cross-Validation
    full_X = np.concatenate([train_X, val_X], axis=0)
    full_y = np.concatenate([train_y, val_y], axis=0)

    print(f"Total training samples: {len(full_y)}")

    # Load Test Data
    test_X, test_y, test_clips = dataset.get_data(
        config.TEST_CSV, "test", load_cached_data=True
    )
    test_dataset = dataset.WhaleDataset(
        test_X, test_y, transform=dataset.get_transforms("test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Storage
    # OOF Predictions: Dictionary of arrays, one for each model_metric combination
    # Test Predictions: Accumulator for bagging
    oof_preds = {}
    test_preds_accum = {}

    # Initialize keys based on architectures and save metrics (auc, loss)
    for arch in config.MODEL_ARCHITECTURES:
        for metric in config.SAVE_METRICS:
            key = f"{arch}_{metric}"
            oof_preds[key] = np.zeros(len(full_y))
            test_preds_accum[key] = np.zeros(len(test_y))

    # 4. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(full_X, full_y)):
        print(f"\n{'='*20} Fold {fold_idx+1}/{config.N_FOLDS} {'='*20}")

        # Prepare DataLoaders for this fold
        X_train_fold, y_train_fold = full_X[train_idx], full_y[train_idx]
        X_val_fold, y_val_fold = full_X[val_idx], full_y[val_idx]

        # Datasets
        train_ds = dataset.WhaleDataset(
            X_train_fold, y_train_fold, transform=dataset.get_transforms("train")
        )
        val_ds = dataset.WhaleDataset(
            X_val_fold, y_val_fold, transform=dataset.get_transforms("val")
        )

        # Sampler for Class Imbalance
        class_counts = np.bincount(y_train_fold)
        # Avoid division by zero
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y_train_fold]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_ds), replacement=True
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            sampler=sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train each Architecture
        for model_name in config.MODEL_ARCHITECTURES:
            # Train and get best metrics (checkpoints are saved inside train_fold)
            # This trains one model but saves TWO checkpoints (Best AUC, Best Loss)
            engine.train_fold(fold_idx, model_name, train_loader, val_loader, device)

            # Load Checkpoints and Generate Predictions (OOF and Test) for both objectives
            for metric in config.SAVE_METRICS:
                key = f"{model_name}_{metric}"
                ckpt_path = os.path.join(
                    config.CHECKPOINT_DIR, f"{model_name}_fold_{fold_idx}_{metric}.pth"
                )

                if not os.path.exists(ckpt_path):
                    print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
                    continue

                # Load Model
                # pretrained=False because we load from local checkpoint
                model = models.get_model(model_name, pretrained=False)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.to(device)
                model.eval()

                # Predict OOF
                val_probs = engine.predict(model, val_loader, device)
                oof_preds[key][val_idx] = val_probs.reshape(-1)

                # Predict Test (Accumulate for Bagging)
                test_probs = engine.predict(model, test_loader, device)
                test_preds_accum[key] += test_probs.reshape(-1)

                # Cleanup to save memory
                del model
                torch.cuda.empty_cache()

    # 5. Average Test Predictions (Bagging)
    test_preds_avg = {k: v / config.N_FOLDS for k, v in test_preds_accum.items()}

    # 6. Meta-Learner Training
    print("\nTraining Meta-Learner...")
    coef, intercept, oof_auc = stacking.train_meta_learner(oof_preds, full_y)

    # 7. Final Validation Metric
    # Printing in the exact format required
    print(f"Final Validation Metric: {oof_auc}")

    # 8. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Meta-Model OOF Probabilities manually to get errors
    X_oof, _ = stacking.format_features(oof_preds)
    logits = np.dot(X_oof, coef) + intercept
    meta_probs = 1.0 / (1.0 + np.exp(-logits))

    errors = np.abs(full_y - meta_probs)

    # Feature for correlation: Mean Spectrogram Intensity
    # full_X shape is (N, 1, F, T)
    mean_intensity = full_X.mean(axis=(1, 2, 3))

    correlation = np.corrcoef(errors, mean_intensity)[0, 1]
    print(
        f"Correlation between Error Magnitude and Mean Spectrogram Intensity: {correlation:.10f}"
    )

    # 9. Submission
    threshold = 0.9998881660199745
    if oof_auc > threshold:
        print(f"\nValidation metric {oof_auc} > {threshold}. Generating submission...")
        final_test_probs = stacking.predict_stack(test_preds_avg, coef, intercept)
        stacking.create_submission(final_test_probs, test_clips)
    else:
        print(
            f"\nValidation metric {oof_auc} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
