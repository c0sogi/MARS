import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratifiedKFold
from scipy.stats import pearsonr
import cv2

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import load_dataset_data, get_transforms, BirdDataset
from library.models import BirdModel
from library.losses import WeightedBCELoss, DistillationLoss, calculate_pos_weights
from library.engine import train_one_epoch, validate, inference


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create submission directory
    os.makedirs("./submission", exist_ok=True)

    # Override Config for Fast Baseline if needed
    # Given the small dataset (206 samples), we can run reasonable epochs.
    # Reducing slightly to ensure 2-hour limit safety for the multi-stage pipeline.
    EPOCHS_STAGE_1 = 15
    EPOCHS_STAGE_2 = 15
    BATCH_SIZE = 32
    N_FOLDS = 5

    # 2. Load Data
    print("Loading Data...")
    # Load Train Data (for CV)
    train_imgs, train_df = load_dataset_data("train", load_cached_data=True)
    # Load Hold-out Validation Data (for Final Metric)
    val_imgs, val_df = load_dataset_data("val", load_cached_data=True)
    # Load Test Data (for Submission)
    test_imgs, test_df = load_dataset_data("test", load_cached_data=True)

    # Extract labels for stratification
    label_cols = [c for c in train_df.columns if c.startswith("species_")]
    y_train = train_df[label_cols].values
    y_val_holdout = val_df[label_cols].values

    # 3. Create Folds
    # We use IterativeStratifiedKFold on the training set
    kfold = IterativeStratifiedKFold(n_splits=N_FOLDS, order=1)
    # X is dummy, we just need indices based on y
    X_dummy = np.zeros((len(y_train), 1))

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_dummy, y_train)):
        folds.append((train_idx, val_idx))

    # Placeholder for OOF Soft Targets (Stage 1 Output)
    # Shape: (N_samples, N_classes)
    oof_soft_targets = np.zeros((len(train_imgs), Config.NUM_CLASSES), dtype=np.float32)
    # Count to average if needed (though standard CV hits each once)
    oof_counts = np.zeros((len(train_imgs), 1), dtype=np.float32)

    # =========================================================================
    # STAGE 1: Train Anchors & Generate OOF
    # =========================================================================
    print(f"\n=== Stage 1: Training Anchors ({Config.ANCHOR_MODELS}) ===")

    # We will store trained anchor weights? No, Stage 2 re-trains them ("Born-Again").
    # We just need the OOF predictions.

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/{N_FOLDS}")

        # Prepare DataLoaders
        train_ds = BirdDataset(
            train_imgs[train_idx],
            train_df.iloc[train_idx],
            transforms=get_transforms("train"),
            phase="train",
        )
        val_ds = BirdDataset(
            train_imgs[val_idx],
            train_df.iloc[val_idx],
            transforms=get_transforms("val"),
            phase="val",
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
        )

        # Calculate Pos Weights for this fold
        fold_labels = train_df.iloc[train_idx][label_cols].values
        pos_weights = calculate_pos_weights(fold_labels).to(device)

        # Train each Anchor
        fold_probs = []  # Store probs from each anchor for this fold

        for model_name in Config.ANCHOR_MODELS:
            model = BirdModel(model_name, pretrained=True).to(device)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            loss_fn = WeightedBCELoss(pos_weights=pos_weights)

            # Train Loop
            for epoch in range(EPOCHS_STAGE_1):
                train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    epoch,
                    loss_fn,
                    mixup_alpha=Config.MIXUP_ALPHA,
                )

            # Predict on Val (OOF)
            model.eval()
            preds = []
            with torch.no_grad():
                for imgs, _, _ in val_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    preds.append(torch.sigmoid(logits).cpu().numpy())

            fold_probs.append(np.concatenate(preds, axis=0))

            # Free memory
            del model, optimizer
            torch.cuda.empty_cache()

        # Average Anchor Predictions for this fold
        avg_preds = np.mean(fold_probs, axis=0)

        # Store in OOF array
        oof_soft_targets[val_idx] = avg_preds
        oof_counts[val_idx] += 1

    # Normalize OOF (should be 1s everywhere for standard CV, but good practice)
    oof_soft_targets = oof_soft_targets / np.maximum(oof_counts, 1)

    # =========================================================================
    # STAGE 2: Distilled Training (Student + Born-Again Anchors)
    # =========================================================================
    print(f"\n=== Stage 2: Distillation Training ===")

    # Models to train: Anchors (Born-Again) + Student
    stage2_models = Config.ANCHOR_MODELS + [Config.STUDENT_MODEL]
    trained_models = []  # Store model objects for inference

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx+1}/{N_FOLDS}")

        # Prepare DataLoaders with Soft Labels
        # Soft labels for training come from OOF (generated in Stage 1)
        # Note: We use the OOF targets corresponding to the training indices
        train_soft = oof_soft_targets[train_idx]
        val_soft = oof_soft_targets[val_idx]  # Not used for loss, but passed to dataset

        train_ds = BirdDataset(
            train_imgs[train_idx],
            train_df.iloc[train_idx],
            transforms=get_transforms("train"),
            soft_labels=train_soft,
            phase="train",
        )
        val_ds = BirdDataset(
            train_imgs[val_idx],
            train_df.iloc[val_idx],
            transforms=get_transforms("val"),
            soft_labels=val_soft,
            phase="val",
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
        )

        # Calculate Pos Weights
        fold_labels = train_df.iloc[train_idx][label_cols].values
        pos_weights = calculate_pos_weights(fold_labels).to(device)

        for model_name in stage2_models:
            print(f"    Training {model_name}...")
            model = BirdModel(model_name, pretrained=True).to(device)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Use Distillation Loss
            loss_fn = DistillationLoss(
                pos_weights=pos_weights,
                lambda_param=Config.DISTILLATION_LAMBDA,
                temperature=Config.DISTILLATION_TEMP,
            )

            best_auc = 0
            best_state = None

            for epoch in range(EPOCHS_STAGE_2):
                train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    epoch,
                    loss_fn,
                    mixup_alpha=Config.MIXUP_ALPHA,
                )

                # Validate (using Hard Labels for metric)
                # We use WeightedBCELoss for validation loss tracking just to be compatible with validate func signature
                # but the metric is what matters.
                val_loss_fn = WeightedBCELoss(pos_weights=pos_weights)
                _, auc = validate(model, val_loader, device, val_loss_fn)

                if auc > best_auc:
                    best_auc = auc
                    best_state = model.state_dict()

            # Load best state
            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            trained_models.append(model)

    # =========================================================================
    # VALIDATION: Hold-out Set Evaluation & Failure Analysis
    # =========================================================================
    print(f"\n=== Final Validation on Hold-out Set ===")

    val_holdout_ds = BirdDataset(
        val_imgs, val_df, transforms=get_transforms("val"), phase="test"
    )
    val_holdout_loader = DataLoader(
        val_holdout_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Ensemble Prediction
    val_preds_accum = []

    for model in trained_models:
        model.eval()
        # Use inference function (TTA) for robust validation score
        preds = inference(model, val_holdout_loader, device)
        val_preds_accum.append(preds)

    # Average predictions
    final_val_preds = np.mean(val_preds_accum, axis=0)

    # Calculate Metric
    final_score = get_score(y_val_holdout, final_val_preds)
    print(f"Final Validation Metric: {final_score}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Mean Squared Error per sample
    mse_per_sample = np.mean((y_val_holdout - final_val_preds) ** 2, axis=1)

    # Compute image features for correlation
    pixel_means = []
    pixel_stds = []
    for img in val_imgs:
        pixel_means.append(np.mean(img))
        pixel_stds.append(np.std(img))

    corr_mean, _ = pearsonr(mse_per_sample, pixel_means)
    corr_std, _ = pearsonr(mse_per_sample, pixel_stds)

    print(f"Correlation (Error vs Pixel Mean): {corr_mean:.4f}")
    print(f"Correlation (Error vs Pixel Std): {corr_std:.4f}")

    # =========================================================================
    # SUBMISSION
    # =========================================================================
    THRESHOLD = 0.9167709334579945

    if final_score > THRESHOLD:
        print(f"\nScore {final_score} > {THRESHOLD}. Generating Submission...")

        test_ds = BirdDataset(
            test_imgs, test_df, transforms=get_transforms("test"), phase="test"
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_preds_accum = []
        for model in trained_models:
            preds = inference(model, test_loader, device)
            test_preds_accum.append(preds)

        final_test_preds = np.mean(test_preds_accum, axis=0)

        # Prepare Submission DataFrame
        # Format: Id, Probability
        # Id = rec_id * 100 + species_id
        submission_rows = []
        rec_ids = test_df["rec_id"].values

        for i, rec_id in enumerate(rec_ids):
            probs = final_test_preds[i]
            for species_idx, prob in enumerate(probs):
                row_id = int(rec_id * 100 + species_idx)
                submission_rows.append({"Id": row_id, "Probability": prob})

        submission_df = pd.DataFrame(submission_rows)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(f"\nScore {final_score} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
