import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_auc
from library.data import BirdDataset, get_transforms, load_dataset_data
from library.models import BirdClassifier
from library.loss import DistillationLoss
from library.engine import train_one_epoch, validate, predict_tta, save_submission


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Overwrite Config for Fast Baseline Execution
    # The prompt requires completion in ~4 minutes.
    # We reduce epochs and folds to the minimum viable demonstration.
    Config.EPOCHS = 3
    Config.N_FOLDS = 2
    Config.BATCH_SIZE = 16  # Reduce batch size for safety on smaller GPUs/memory

    print(f"Starting Born-Again Ensemble Pipeline on {device}...")

    # 2. Load Training Data
    # We load the entire development set (Fold 0 from original split, which is 'train.csv')
    train_images, train_labels, train_ids = load_dataset_data(
        "train", load_cached_data=True
    )

    # Calculate pos_weight for loss stability
    # shape: (num_classes,)
    num_pos = np.sum(train_labels, axis=0)
    num_neg = len(train_labels) - num_pos
    # Clip to avoid division by zero or extreme weights
    pos_weight = torch.tensor(
        np.clip(num_neg / (num_pos + 1e-6), 1.0, 20.0), dtype=torch.float32
    ).to(device)

    # Container for Soft Targets (Phase 2 artifact)
    # We will fill this during Phase 1 OOF predictions
    soft_targets_all = np.zeros_like(train_labels)

    # K-Fold Splitter
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # =========================================================================
    # Phase 1: Anchor Training & TTA-Target Generation
    # =========================================================================
    print("\n=== Phase 1: Anchor Training & Target Generation ===")

    phase1_models_store = (
        []
    )  # We won't strictly need to store them for Phase 3 inference, but good for tracking

    # We iterate through folds to generate OOF soft targets for the entire train set
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f"  Fold {fold+1}/{Config.N_FOLDS}")

        # Split data
        X_train, y_train = train_images[train_idx], train_labels[train_idx]
        X_val, y_val = train_images[val_idx], train_labels[val_idx]

        # Create Datasets
        train_ds = BirdDataset(
            X_train, labels=y_train, transforms=get_transforms("train")
        )
        val_ds = BirdDataset(
            X_val, labels=y_val, transforms=get_transforms("val")
        )  # Standard val transform for checking

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train each Anchor Model
        fold_anchors = []
        for model_name in Config.PHASE1_MODELS:
            # Initialize Model
            model = BirdClassifier(
                model_name, num_classes=Config.NUM_CLASSES, pretrained=True
            ).to(device)
            optimizer = optim.AdamW(
                model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
            )
            criterion = DistillationLoss(
                pos_weight=pos_weight
            )  # Standard BCE mode (soft_targets=None)

            # Training Loop
            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device
                )

            # We don't do extensive validation here to save time, just store the model
            fold_anchors.append(model)

        # Generate TTA-Enhanced Soft Targets for the Validation part of this fold
        # We use the trained anchors to predict on X_val
        # Note: We use predict_tta which applies cyclic shifts and averages them
        # This creates the "Smooth, Invariant" teacher signal
        print(f"    Generating Soft Targets for fold {fold+1}...")
        oof_probs = predict_tta(fold_anchors, X_val, device)

        # Store in the global soft_targets array
        soft_targets_all[val_idx] = oof_probs

        # Clean up to save memory
        del fold_anchors, model, optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache()

    # =========================================================================
    # Phase 3: Born-Again Ensemble Training
    # =========================================================================
    print("\n=== Phase 3: Born-Again Ensemble Training ===")

    final_ensemble = []

    # We re-train using the generated soft targets
    # We use the same folds to ensure we don't leak (though with Born-Again on full data,
    # leakage is less of a concern if we trust the teacher. Here we stick to fold structure for robustness).

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f"  Fold {fold+1}/{Config.N_FOLDS}")

        # Input Data
        X_train = train_images[train_idx]
        y_train_hard = train_labels[train_idx]
        y_train_soft = soft_targets_all[train_idx]  # The distilled knowledge

        # Dataset with Soft Labels
        train_ds = BirdDataset(
            X_train,
            labels=y_train_hard,
            soft_labels=y_train_soft,
            transforms=get_transforms("train"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train all models in the final ensemble (ResNet, EfficientNet, DenseNet)
        for model_name in Config.PHASE3_MODELS:
            model = BirdClassifier(
                model_name, num_classes=Config.NUM_CLASSES, pretrained=True
            ).to(device)
            optimizer = optim.AdamW(
                model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
            )

            # Loss with Distillation enabled
            # We pass pos_weight to the DistillationLoss class
            criterion = DistillationLoss(
                pos_weight=pos_weight, distillation_lambda=Config.DISTILLATION_LAMBDA
            )

            for epoch in range(Config.EPOCHS):
                # train_one_epoch automatically handles soft_target if present in batch
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device
                )

            final_ensemble.append(model)

            # Save model (optional, but good practice)
            # torch.save(model.state_dict(), os.path.join(Config.MODEL_DIR, f"{model_name}_fold{fold}.pth"))

    # =========================================================================
    # 4. Validation & Failure Analysis
    # =========================================================================
    print("\n=== Validation ===")

    # Load Hold-out Validation Set
    val_images, val_labels, val_ids = load_dataset_data("val", load_cached_data=True)

    # Inference on Validation Set using the Full Ensemble and TTA
    # This matches the submission pipeline
    val_probs = predict_tta(final_ensemble, val_images, device)

    # Compute Metric
    val_auc = compute_auc(val_labels, val_probs)
    print(f"Final Validation Metric: {val_auc:.16f}")

    # Failure Analysis
    # Calculate per-sample error (Mean Absolute Error between pred and true)
    # and correlate with number of labels (Label Cardinality)
    abs_error = np.abs(val_labels - val_probs).mean(axis=1)
    label_cardinality = val_labels.sum(axis=1)

    # Simple correlation
    if np.std(abs_error) > 0 and np.std(label_cardinality) > 0:
        correlation = np.corrcoef(abs_error, label_cardinality)[0, 1]
    else:
        correlation = 0.0

    print(f"Failure Analysis - Correlation (Error vs Label Count): {correlation:.4f}")
    print(
        "Interpretation: Positive correlation implies model struggles with multi-species recordings."
    )

    # =========================================================================
    # 5. Submission
    # =========================================================================
    # Threshold from prompt
    THRESHOLD = 0.92133638985917

    if val_auc > THRESHOLD:
        print("\n=== Generating Submission ===")

        # Load Test Data
        test_images, _, test_ids = load_dataset_data("test", load_cached_data=True)

        # Inference
        test_probs = predict_tta(final_ensemble, test_images, device)

        # Save
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        save_submission(test_ids, test_probs, sub_path)
    else:
        print(
            f"\nValidation metric {val_auc} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
