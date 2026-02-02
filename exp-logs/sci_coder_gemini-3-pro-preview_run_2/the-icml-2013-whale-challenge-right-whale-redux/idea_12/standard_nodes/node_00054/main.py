import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, save_checkpoint
from library.dataset import WhaleDataset, get_transforms, process_and_cache_data
from library.models import WhaleClassifier
from library.engine import train_fn, eval_fn, inference_fn

# Set seeds for reproducibility
seed_everything(Config.SEED)


def create_loader(
    data, labels=None, clips=None, mode="train", batch_size=Config.BATCH_SIZE
):
    """
    Helper to create DataLoader with appropriate sampling and transforms.
    """
    transform = get_transforms(mode) if mode == "train" else None
    dataset = WhaleDataset(data, labels=labels, clips=clips, transform=transform)

    sampler = None
    shuffle = False

    if mode == "train" and labels is not None:
        # Weighted Random Sampler to handle class imbalance
        class_counts = np.bincount(labels)
        # Avoid division by zero
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
    elif mode == "train":
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(mode == "train"),
    )

    return loader


def run():
    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Override Config for Fast Baseline Execution
    EPOCHS = 10  # Increased to ensure convergence
    PATIENCE = 4

    # ------------------------------------------------------------------
    # 2. Data Preparation (Train/Val Split for CV)
    # ------------------------------------------------------------------
    print("Loading Training Data...")
    # This loads the 'train.csv' data (80% of original labeled data)
    train_data_full, train_labels_full, _ = process_and_cache_data(
        "train", load_cached_data=True
    )

    # We will perform 5-Fold Stratified CV on this training set
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for Out-Of-Fold (OOF) predictions
    # Shape: (N_samples, N_models) -> We have 2 models in the ensemble
    oof_preds = np.zeros((len(train_labels_full), len(Config.MODELS)))

    # Store trained model paths for inference later
    trained_model_paths = {model_name: [] for model_name in Config.MODELS}

    # ------------------------------------------------------------------
    # 3. Level 0: Train Base Models (5-Fold CV)
    # ------------------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_data_full, train_labels_full)
    ):
        print(f"\n=== Fold {fold + 1}/{Config.N_FOLDS} ===")

        # Slice data for this fold
        X_train, y_train = train_data_full[train_idx], train_labels_full[train_idx]
        X_val, y_val = train_data_full[val_idx], train_labels_full[val_idx]

        # Create Loaders
        train_loader = create_loader(X_train, y_train, mode="train")
        val_loader = create_loader(X_val, y_val, mode="val")

        # Train each model architecture
        for model_idx, model_name in enumerate(Config.MODELS):
            print(f"Training {model_name}...")

            # Initialize Model
            model = WhaleClassifier(model_name, pretrained=Config.USE_PRETRAINED)
            model.to(device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS, eta_min=Config.ETA_MIN
            )
            criterion = nn.BCEWithLogitsLoss()

            # Save Path
            save_filename = f"{model_name}_fold_{fold}.pth"
            save_path = os.path.join(Config.WORKING_DIR, save_filename)
            trained_model_paths[model_name].append(save_path)

            # Train
            train_fn(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                criterion,
                epochs=EPOCHS,
                patience=PATIENCE,
                save_path=save_path,
            )

            # Load Best Model for OOF Prediction
            checkpoint = torch.load(save_path, map_location=device)
            model.load_state_dict(checkpoint)

            # Generate OOF Predictions
            _, _, preds = eval_fn(model, val_loader, device, criterion)

            # Store OOF (align with original indices)
            oof_preds[val_idx, model_idx] = preds

    # ------------------------------------------------------------------
    # 4. Level 1: Train Meta-Learner
    # ------------------------------------------------------------------
    print("\nTraining Meta-Learner (Logistic Regression)...")
    meta_learner = LogisticRegression()
    meta_learner.fit(oof_preds, train_labels_full)

    # Check Meta-Learner Performance on OOF
    meta_oof_preds = meta_learner.predict_proba(oof_preds)[:, 1]
    oof_auc = calculate_roc_auc(train_labels_full, meta_oof_preds)
    print(f"Level 1 OOF AUC: {oof_auc:.5f}")
    print(f"Meta-Learner Coefficients: {meta_learner.coef_}")

    # ------------------------------------------------------------------
    # 5. Final Validation (Hold-out Set)
    # ------------------------------------------------------------------
    print("\nRunning Final Validation on Hold-out Set...")
    # Load Hold-out Validation Data (from val.csv)
    val_data_holdout, val_labels_holdout, _ = process_and_cache_data(
        "val", load_cached_data=True
    )
    val_loader_holdout = create_loader(val_data_holdout, val_labels_holdout, mode="val")

    # Generate Predictions from Base Models
    # We average predictions across the 5 folds for each architecture

    # Shape: (N_val, N_models)
    base_model_val_preds = np.zeros((len(val_labels_holdout), len(Config.MODELS)))

    for model_idx, model_name in enumerate(Config.MODELS):
        fold_preds = []
        for fold_path in trained_model_paths[model_name]:
            # Load Model
            model = WhaleClassifier(
                model_name, pretrained=False
            )  # Weights loaded from checkpoint
            checkpoint = torch.load(fold_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()

            # Predict
            _, _, preds = eval_fn(
                model, val_loader_holdout, device, nn.BCEWithLogitsLoss()
            )
            fold_preds.append(preds)

        # Average across folds
        avg_preds = np.mean(fold_preds, axis=0)
        base_model_val_preds[:, model_idx] = avg_preds

    # Meta-Learner Prediction
    final_val_probs = meta_learner.predict_proba(base_model_val_preds)[:, 1]

    # Compute Metric
    final_val_auc = calculate_roc_auc(val_labels_holdout, final_val_probs)
    print(f"Final Validation Metric: {final_val_auc}")

    # ------------------------------------------------------------------
    # 6. Failure Analysis
    # ------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    # Calculate Error Magnitude
    errors = np.abs(val_labels_holdout - final_val_probs)

    # Extract simple features from spectrograms for correlation
    # Flatten spatial dims for stats: (N, F*T)
    flat_specs = val_data_holdout.reshape(val_data_holdout.shape[0], -1)

    feat_mean = np.mean(flat_specs, axis=1)
    feat_std = np.std(flat_specs, axis=1)
    feat_max = np.max(flat_specs, axis=1)

    # Correlation
    corr_mean = np.corrcoef(errors, feat_mean)[0, 1]
    corr_std = np.corrcoef(errors, feat_std)[0, 1]
    corr_max = np.corrcoef(errors, feat_max)[0, 1]

    print(f"Correlation between Error and Spectrogram Mean: {corr_mean:.4f}")
    print(f"Correlation between Error and Spectrogram Std: {corr_std:.4f}")
    print(f"Correlation between Error and Spectrogram Max: {corr_max:.4f}")

    # ------------------------------------------------------------------
    # 7. Submission
    # ------------------------------------------------------------------
    THRESHOLD = 0.9959928858461402

    if final_val_auc > THRESHOLD:
        print("\nValidation metric met threshold. Generating submission...")

        # Load Test Data
        test_data, _, test_clips = process_and_cache_data("test", load_cached_data=True)
        test_loader = create_loader(test_data, clips=test_clips, mode="test")

        # Base Model Predictions (Average over folds)
        base_model_test_preds = np.zeros((len(test_data), len(Config.MODELS)))

        for model_idx, model_name in enumerate(Config.MODELS):
            fold_preds = []
            for fold_path in trained_model_paths[model_name]:
                model = WhaleClassifier(model_name, pretrained=False)
                checkpoint = torch.load(fold_path, map_location=device)
                model.load_state_dict(checkpoint)
                model.to(device)

                # Inference
                _, preds = inference_fn(model, test_loader, device)
                fold_preds.append(preds)

            avg_preds = np.mean(fold_preds, axis=0)
            base_model_test_preds[:, model_idx] = avg_preds

        # Meta-Learner Prediction
        final_test_probs = meta_learner.predict_proba(base_model_test_preds)[:, 1]

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"clip": test_clips, "probability": final_test_probs})

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {final_val_auc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
