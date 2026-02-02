import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import library components
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    load_checkpoint,
    AverageMeter,
)
from library.models import get_model
from library.dataset import load_dataset_data, WhaleDataset
from library.train import run_fold, inference as run_submission_inference

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Reduce epochs to ensure execution within 2 hours limit.
# 10 models * 4 epochs * ~2 mins/epoch = ~80 mins, leaving buffer for inference.
Config.EPOCHS = 4


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading (Train Only)
    # We strictly use train.csv for training to preserve val.csv as a pure hold-out set
    print("Loading training data...")
    train_data, train_labels, _ = load_dataset_data(Config.TRAIN_CSV, "train")

    # 3. Training Loop
    # We implement the CV loop manually to ensure we only train on train.csv
    skf = StratifiedKFold(n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED)

    # Iterate over architectures (Heterogeneous Ensemble)
    for model_name in Config.MODEL_NAMES:
        print(f"\n{'='*40}")
        print(f"Training Architecture: {model_name}")
        print(f"{'='*40}")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_data, train_labels)
        ):
            print(f"\n--- Fold {fold}/{Config.FOLDS} ---")

            # Split Data
            X_train, X_val = train_data[train_idx], train_data[val_idx]
            y_train, y_val = train_labels[train_idx], train_labels[val_idx]

            # Create Datasets
            train_ds = WhaleDataset(X_train, y_train, is_training=True)
            val_ds = WhaleDataset(X_val, y_val, is_training=False)

            # Weighted Sampler for Class Balance
            class_counts = np.bincount(y_train.astype(int))
            # Handle potential edge cases in small subsets
            if len(class_counts) < 2:
                weights = np.ones(2)
            else:
                weights = 1.0 / (class_counts + 1e-6)

            sample_weights = [weights[int(y)] for y in y_train]
            sampler = WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True
            )

            # Create Loaders
            train_loader = DataLoader(
                train_ds,
                batch_size=Config.BATCH_SIZE,
                sampler=sampler,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Run Training for this fold
            # This saves the model to Config.OUTPUT_DIR
            run_fold(fold, model_name, train_loader, val_loader, device)

    # 4. Validation on Hold-Out Set
    print(f"\n{'='*40}")
    print("Evaluating on Hold-Out Validation Set")
    print(f"{'='*40}")

    val_data_holdout, val_labels_holdout, _ = load_dataset_data(Config.VAL_CSV, "val")
    val_ds_holdout = WhaleDataset(
        val_data_holdout, val_labels_holdout, is_training=False
    )
    val_loader_holdout = DataLoader(
        val_ds_holdout,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensemble Inference
    final_preds = np.zeros(len(val_labels_holdout))
    model_count = 0

    for model_name in Config.MODEL_NAMES:
        for fold in range(Config.FOLDS):
            filename = f"{model_name}_fold_{fold}.pth"
            try:
                # Load Model
                model = get_model(model_name, pretrained=False)
                load_checkpoint(model, filename, device=device)
                model.to(device)
                model.eval()

                fold_preds = []
                with torch.no_grad():
                    for images, _ in val_loader_holdout:
                        images = images.to(device)
                        logits = model(images)
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                        fold_preds.extend(probs)

                final_preds += np.array(fold_preds)
                model_count += 1

                # Clean up to save memory
                del model
                torch.cuda.empty_cache()

            except FileNotFoundError:
                print(f"Warning: Checkpoint {filename} not found. Skipping.")

    if model_count == 0:
        print("Error: No models available for validation.")
        return

    # Soft Voting
    final_preds /= model_count

    # Calculate Metric
    val_auc = calculate_roc_auc(val_labels_holdout, final_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print(f"\n{'='*40}")
    print("Failure Analysis")
    print(f"{'='*40}")

    errors = np.abs(val_labels_holdout - final_preds)

    # Extract simple features from waveforms for correlation analysis
    durations = []
    rms_values = []
    peak_values = []

    for wav in val_data_holdout:
        durations.append(len(wav) / Config.SAMPLE_RATE)
        rms = np.sqrt(np.mean(wav**2))
        rms_values.append(rms)
        peak_values.append(np.max(np.abs(wav)))

    features = {
        "Duration": np.array(durations),
        "RMS": np.array(rms_values),
        "Peak": np.array(peak_values),
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, feat_vals in features.items():
        if np.std(feat_vals) > 1e-9:
            # Use numpy for correlation
            corr = np.corrcoef(errors, feat_vals)[0, 1]
            print(f"{name}: {corr:.4f}")
        else:
            print(f"{name}: N/A (Constant)")

    # 6. Submission
    THRESHOLD = 0.9959177895986835
    if val_auc > THRESHOLD:
        print(
            f"\nValidation Metric ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        run_submission_inference()
    else:
        print(
            f"\nValidation Metric ({val_auc}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
