import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from scipy.stats import pearsonr

# Import from provided library
from library import config, utils, data, models, engine, losses


def run_pipeline():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # Override config for fast baseline execution
    # The dataset is small, so 15 epochs is sufficient for convergence
    config.EPOCHS = 15

    # Models to use in the heterogeneous ensemble
    model_names = ["resnet18", "efficientnet_b0", "densenet121"]

    # =========================================================================
    # STAGE 1: Teacher Generation (Standard Supervised)
    # =========================================================================
    print("\n=== STAGE 1: Teacher Generation ===")

    # Dictionary to store OOF predictions: rec_id -> list of prob arrays
    oof_preds_accumulator = {}

    # Initialize accumulator for all train IDs
    df_train_all = pd.read_csv(config.TRAIN_CSV)
    all_train_ids = df_train_all["rec_id"].values
    for rid in all_train_ids:
        oof_preds_accumulator[rid] = []

    # Iterate over folds
    for fold in range(config.N_FOLDS):
        print(f"  Fold {fold}/{config.N_FOLDS - 1}")

        for model_name in model_names:
            # Create Loaders (Stage 1: No soft labels)
            train_loader, val_loader = data.create_loaders(
                fold=fold, model_name=model_name, soft_labels=None
            )

            # Initialize Model
            model = models.BirdClassifier(
                model_name=model_name, num_classes=config.NUM_SPECIES
            ).to(device)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.EPOCHS
            )

            # Loss Function (Standard Supervised: Gamma=1.0)
            loss_fn = losses.DistillationLoss(gamma=1.0)

            # Training Loop
            for epoch in range(config.EPOCHS):
                _ = engine.train_one_epoch(
                    model, train_loader, optimizer, device, loss_fn, scheduler=None
                )
                if scheduler:
                    scheduler.step()

            # Generate OOF Predictions for this fold's validation set using TTA
            fold_preds = engine.predict(model, val_loader, device)

            # Store OOFs
            for rid, prob in fold_preds.items():
                oof_preds_accumulator[rid].append(prob)

            # Clean up to save memory
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

    # Aggregate OOFs to create Soft Labels
    print("  Aggregating OOF Predictions...")
    soft_labels = {}
    for rid, prob_list in oof_preds_accumulator.items():
        if len(prob_list) > 0:
            # Average probabilities across all models that predicted this ID
            soft_labels[rid] = np.mean(np.stack(prob_list), axis=0)
        else:
            soft_labels[rid] = np.zeros(config.NUM_SPECIES)

    # =========================================================================
    # STAGE 2: Student Distillation
    # =========================================================================
    print("\n=== STAGE 2: Student Distillation ===")

    stage2_models = []

    # We group models by name to facilitate batched inference later
    models_by_name = {name: [] for name in model_names}

    for fold in range(config.N_FOLDS):
        print(f"  Fold {fold}/{config.N_FOLDS - 1}")

        for model_name in model_names:
            # Create Loaders (Stage 2: With Soft Labels)
            train_loader, val_loader = data.create_loaders(
                fold=fold, model_name=model_name, soft_labels=soft_labels
            )

            model = models.BirdClassifier(
                model_name=model_name, num_classes=config.NUM_SPECIES
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.EPOCHS
            )

            # Loss Function (Distillation: Gamma=0.5)
            loss_fn = losses.DistillationLoss(gamma=config.DISTILLATION_GAMMA)

            # Training
            for epoch in range(config.EPOCHS):
                _ = engine.train_one_epoch(
                    model, train_loader, optimizer, device, loss_fn, scheduler=None
                )
                if scheduler:
                    scheduler.step()

            # Store model for final inference
            model.eval()
            models_by_name[model_name].append(model)
            stage2_models.append(model)

            del train_loader, val_loader, optimizer, scheduler
            torch.cuda.empty_cache()

    # =========================================================================
    # EVALUATION ON HOLD-OUT VALIDATION SET
    # =========================================================================
    print("\n=== Evaluation on Hold-out Validation Set ===")

    # Load Validation Set
    df_val = pd.read_csv(config.VAL_CSV)
    val_preds_accumulator = {rid: [] for rid in df_val["rec_id"].values}

    # Inference using all Stage 2 models
    for model_name, model_list in models_by_name.items():
        # Create a dataset/loader for this model type (for image size/transforms)
        val_dataset = data.BirdDataset(
            df_val, transforms=data.get_transforms(model_name, mode="val"), mode="val"
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

        for model in model_list:
            # Predict with TTA
            preds = engine.predict(model, val_loader, device)
            for rid, prob in preds.items():
                val_preds_accumulator[rid].append(prob)

    # Aggregate Predictions
    final_val_preds = []
    final_val_targets = []
    label_cols = [c for c in df_val.columns if c.startswith("species_")]

    for idx, row in df_val.iterrows():
        rid = row["rec_id"]
        probs_list = val_preds_accumulator[rid]
        # Average across all ensemble members
        avg_prob = np.mean(np.stack(probs_list), axis=0)
        final_val_preds.append(avg_prob)
        final_val_targets.append(row[label_cols].values.astype(float))

    final_val_preds = np.array(final_val_preds)
    final_val_targets = np.array(final_val_targets)

    # Compute Metric
    val_score = utils.get_score(final_val_targets, final_val_preds)
    print(f"Final Validation Metric: {val_score}")

    # =========================================================================
    # FAILURE ANALYSIS
    # =========================================================================
    print("\n=== Failure Analysis ===")
    # Calculate Mean Absolute Error per sample
    mae_per_sample = np.mean(np.abs(final_val_preds - final_val_targets), axis=1)

    # Compute spectrogram stats for correlation
    pixel_means = []
    pixel_stds = []

    for idx, row in df_val.iterrows():
        # Reconstruct path logic to match BirdDataset
        fname = os.path.basename(row["file_path_spec"])
        full_path = os.path.join(config.SPECTROGRAM_DIR, fname)

        if os.path.exists(full_path):
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                pixel_means.append(np.mean(img))
                pixel_stds.append(np.std(img))
            else:
                pixel_means.append(0)
                pixel_stds.append(0)
        else:
            pixel_means.append(0)
            pixel_stds.append(0)

    # Correlation
    if len(pixel_means) == len(mae_per_sample):
        corr_mean, _ = pearsonr(pixel_means, mae_per_sample)
        corr_std, _ = pearsonr(pixel_stds, mae_per_sample)
        print(f"Correlation (Pixel Mean vs Error): {corr_mean:.4f}")
        print(f"Correlation (Pixel Std vs Error): {corr_std:.4f}")

    # =========================================================================
    # SUBMISSION
    # =========================================================================
    threshold = 0.9129501920716607

    if val_score > threshold:
        print("\n=== Generating Submission ===")

        # Load Test Set
        df_test = pd.read_csv(config.TEST_CSV)
        test_preds_accumulator = {rid: [] for rid in df_test["rec_id"].values}

        for model_name, model_list in models_by_name.items():
            # Create Test Loader
            test_loader = data.create_test_loader(
                model_name, batch_size=config.BATCH_SIZE
            )

            for model in model_list:
                preds = engine.predict(model, test_loader, device)
                for rid, prob in preds.items():
                    test_preds_accumulator[rid].append(prob)

        # Format Submission
        submission_rows = []
        # Ensure we iterate in a deterministic order or just iterate through dataframe
        # The sample submission requires specific IDs.
        # We iterate through all rec_ids present in test set.

        for rid in sorted(test_preds_accumulator.keys()):
            probs_list = test_preds_accumulator[rid]
            avg_prob = np.mean(np.stack(probs_list), axis=0)  # Shape (19,)

            # Format: Id,Probability
            # Id = rec_id * 100 + species_id
            for species_id, prob in enumerate(avg_prob):
                submission_id = int(rid * 100 + species_id)
                submission_rows.append([submission_id, prob])

        df_sub = pd.DataFrame(submission_rows, columns=["Id", "Probability"])
        df_sub.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {val_score} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
