import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metrics
from library.data import make_folds, get_dataloaders, get_test_loader
from library.models import BirdModel
from library.engine import (
    train_one_epoch,
    validate,
    inference_with_tta,
    save_submission,
    get_pos_weight,
)


def main():
    # 1. Setup and Configuration
    # Adjust MAX_STEPS for the fast baseline constraint while ensuring convergence
    Config.MAX_STEPS = 600
    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Prepare Data (Folds)
    # This creates/loads the 5-fold split dataframe
    df_folds = make_folds(load_cached_data=True)

    # Dictionary to store Out-Of-Fold (OOF) predictions
    # Key: rec_id, Value: list of probability arrays (one per model in the ensemble)
    oof_preds_dict = {}
    oof_targets_dict = {}

    # Store trained model info for Test Inference later
    # Tuple: (backbone_name, model_path, img_size)
    trained_models = []

    # 3. Cross-Validation Loop
    for fold in range(Config.NUM_FOLDS):
        print(f"\n=== Starting Fold {fold}/{Config.NUM_FOLDS - 1} ===")

        # Prepare class weights for this fold to handle imbalance
        train_df_fold = df_folds[df_folds["fold"] != fold]
        pos_weight = get_pos_weight(train_df_fold).to(device)

        # Heterogeneous Ensemble: Train each backbone
        for backbone in Config.BACKBONES:
            print(f"Training Backbone: {backbone}")

            # Determine input resolution
            img_size = Config.get_image_size(backbone)

            # Get DataLoaders
            train_loader, val_loader = get_dataloaders(
                fold_idx=fold, img_size=img_size, batch_size=Config.BATCH_SIZE
            )

            # Initialize Model
            model = BirdModel(backbone_name=backbone, pretrained=True).to(device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            # Constant LR, so no scheduler needed (or trivial one)

            # Training Loop
            step_count = 0
            epoch = 0
            best_auc = -1.0
            model_save_path = os.path.join(
                Config.WORKING_DIR, f"{backbone}_fold{fold}.pth"
            )

            while step_count < Config.MAX_STEPS:
                epoch += 1
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, device, pos_weight=pos_weight
                )
                step_count += len(train_loader)

                # Validation
                # Check at least once at the end, or periodically
                if epoch % 5 == 0 or step_count >= Config.MAX_STEPS:
                    val_loss, val_auc = validate(model, val_loader, device, pos_weight)

                    if val_auc > best_auc:
                        best_auc = val_auc
                        torch.save(model.state_dict(), model_save_path)

            print(f"  Finished {backbone} Fold {fold}. Best AUC: {best_auc:.4f}")
            trained_models.append((backbone, model_save_path, img_size))

            # Generate OOF Predictions with Best Model
            model.load_state_dict(torch.load(model_save_path))
            model.eval()

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    rec_ids = batch["rec_id"].numpy()
                    labels = batch["labels"].numpy()

                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy()

                    for i, rid in enumerate(rec_ids):
                        if rid not in oof_preds_dict:
                            oof_preds_dict[rid] = []
                            oof_targets_dict[rid] = labels[i]
                        oof_preds_dict[rid].append(probs[i])

    # 4. Grand Ensemble & Metric Calculation
    print("\n=== Computing Final Validation Metrics ===")

    final_preds = []
    final_targets = []

    # Sort by rec_id to ensure alignment
    sorted_rec_ids = sorted(oof_preds_dict.keys())

    for rid in sorted_rec_ids:
        # Average predictions from all models for this recording
        # (Should be 3 models per fold)
        model_preds = np.array(oof_preds_dict[rid])
        avg_pred = np.mean(model_preds, axis=0)

        final_preds.append(avg_pred)
        final_targets.append(oof_targets_dict[rid])

    final_preds = np.array(final_preds)
    final_targets = np.array(final_targets)

    val_metric = calculate_metrics(final_targets, final_preds)
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Mean Absolute Error per sample
    mae_per_sample = np.mean(np.abs(final_targets - final_preds), axis=1)

    # Feature 1: Number of Labels (Complexity)
    num_labels = np.sum(final_targets, axis=1)

    # Feature 2: 'Has Bird' (Binary)
    has_bird = (num_labels > 0).astype(int)

    # Correlations
    if np.std(mae_per_sample) > 0 and np.std(num_labels) > 0:
        corr_complexity = np.corrcoef(mae_per_sample, num_labels)[0, 1]
    else:
        corr_complexity = 0.0

    print(f"Correlation (Error vs Label Count): {corr_complexity}")

    # 6. Test Inference & Submission
    THRESHOLD = 0.9129501920716607

    if val_metric > THRESHOLD:
        print(
            f"\nMetric ({val_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_accum = {}  # rec_id -> list of prob arrays

        for backbone, path, img_size in trained_models:
            # Load Model
            model = BirdModel(backbone_name=backbone, pretrained=False).to(device)
            model.load_state_dict(torch.load(path))
            model.eval()

            # Get Test Loader (specific to image size)
            test_loader = get_test_loader(
                img_size=img_size, batch_size=Config.BATCH_SIZE
            )

            # Inference with TTA
            rec_ids, probs = inference_with_tta(model, test_loader, device)

            # Accumulate
            for i, rid in enumerate(rec_ids):
                if rid not in test_accum:
                    test_accum[rid] = []
                test_accum[rid].append(probs[i])

        # Average and Save
        final_test_ids = []
        final_test_probs = []

        for rid in sorted(test_accum.keys()):
            avg_p = np.mean(test_accum[rid], axis=0)
            final_test_ids.append(rid)
            final_test_probs.append(avg_p)

        save_submission(
            np.array(final_test_ids), np.array(final_test_probs), Config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nMetric ({val_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
