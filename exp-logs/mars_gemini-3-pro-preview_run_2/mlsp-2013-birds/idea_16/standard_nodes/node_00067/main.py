import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
import warnings
from skmultilearn.model_selection import IterativeStratifiedKFold

# Import library modules
from library.utils import seed_everything, get_device, calculate_metrics
from library.dataset import get_loader, load_and_cache_images
from library.models import BirdModel
from library.trainer import train_model
from library.inference import run_inference, TTADataset, predict_with_model, TTA_SHIFTS
from torch.utils.data import DataLoader

# Configuration
SEED = 42
BATCH_SIZE = 16  # Reduced batch size (Cite 00042)
NUM_EPOCHS = 60  # Increased epochs to ensure sufficient steps (Cite 00031)
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ARCHITECTURES = ["resnet18", "densenet121"]  # Removed EfficientNet (Cite 00042)
WORKING_DIR = "./working/idea_16"
METADATA_DIR = "./metadata"
INPUT_DIR = "./input"
THRESHOLD = 0.9129501920716607
FOLDS = 5


def evaluate_ensemble_on_val(df_val, models_dict, device):
    """
    Evaluates the ensemble on the validation set using TTA.
    models_dict: {arch_name: model_instance}
    """
    # Initialize aggregated predictions
    # Shape: (N_samples, N_classes)
    num_samples = len(df_val)
    num_classes = 19

    # We need to map rec_id to index to aggregate correctly
    rec_ids = df_val["rec_id"].values
    id_to_idx = {rid: i for i, rid in enumerate(rec_ids)}

    ensemble_probs = np.zeros((num_samples, num_classes), dtype=np.float32)

    # Pre-load images for TTA dataset
    # We use a dummy arch name to get cache, resolution handled inside loop
    image_dict = load_and_cache_images(df_val, WORKING_DIR, load_cached_data=True)

    total_passes = 0

    for arch, model in models_dict.items():
        # Determine resolution
        if "densenet" in arch:
            height, width = 160, 320
        else:
            height, width = 224, 448

        for shift in TTA_SHIFTS:
            # Create TTA Dataset
            dataset = TTADataset(df_val, image_dict, height, width, shift_pct=shift)
            loader = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
            )

            # Predict
            preds_dict = predict_with_model(model, loader, device)

            # Accumulate
            for rid, probs in preds_dict.items():
                if rid in id_to_idx:
                    idx = id_to_idx[rid]
                    ensemble_probs[idx] += probs

            total_passes += 1

    # Average
    ensemble_probs /= total_passes
    return ensemble_probs


def perform_failure_analysis(df_val, y_true, y_pred, image_dict):
    """
    Correlates error with image statistics.
    """
    # Calculate Mean Absolute Error per sample
    # y_true and y_pred are (N, 19)
    # error per sample: mean over classes
    errors = np.mean(np.abs(y_true - y_pred), axis=1)

    # Calculate Image Stats
    pixel_means = []
    pixel_stds = []

    for rid in df_val["rec_id"].values:
        if rid in image_dict:
            img = image_dict[rid]
            pixel_means.append(np.mean(img))
            pixel_stds.append(np.std(img))
        else:
            pixel_means.append(0)
            pixel_stds.append(0)

    # Correlation
    corr_mean, _ = pearsonr(errors, pixel_means)
    corr_std, _ = pearsonr(errors, pixel_stds)

    print("\nFailure Analysis:")
    print(f"Correlation (Error vs Pixel Mean): {corr_mean:.4f}")
    print(f"Correlation (Error vs Pixel Std): {corr_std:.4f}")
    print(
        "Interpretation: Positive correlation implies higher error on brighter/higher-contrast images."
    )


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    os.makedirs(WORKING_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    print(f"Using device: {device}")

    # 2. Load Data
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        print(
            "Metadata not found. Please ensure ./metadata/train.csv and val.csv exist."
        )
        return

    df_train_part = pd.read_csv(train_csv_path)
    df_val_part = pd.read_csv(val_csv_path)

    # Combine for Cross-Validation (Cite 00018, 00061)
    df_full = pd.concat([df_train_part, df_val_part]).reset_index(drop=True)
    print(f"Total development samples: {len(df_full)}")

    # Identify label columns
    label_cols = [c for c in df_full.columns if c.startswith("species_")]
    num_classes = len(label_cols)

    # Prepare for Stratified K-Fold
    X = df_full["rec_id"].values.reshape(-1, 1)
    y = df_full[label_cols].values

    kfold = IterativeStratifiedKFold(n_splits=FOLDS, order=1, random_state=SEED)

    # Store OOF predictions
    oof_preds = np.zeros((len(df_full), num_classes))
    oof_targets = np.zeros((len(df_full), num_classes))

    # Map rec_id to row index in df_full for OOF filling
    id_to_idx = {rid: i for i, rid in enumerate(df_full["rec_id"].values)}

    # 3. CV Loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n=== Fold {fold} ===")

        df_train_fold = df_full.iloc[train_idx].copy()
        df_val_fold = df_full.iloc[val_idx].copy()

        trained_models_fold = {}

        for arch in ARCHITECTURES:
            print(f"Training Architecture: {arch}")

            # Get Loaders
            train_loader = get_loader(
                df_train_fold,
                arch,
                phase="train",
                batch_size=BATCH_SIZE,
                num_workers=2,
                load_cached_data=True,
                cache_dir=WORKING_DIR,
            )
            val_loader = get_loader(
                df_val_fold,
                arch,
                phase="val",
                batch_size=BATCH_SIZE,
                num_workers=2,
                load_cached_data=True,
                cache_dir=WORKING_DIR,
            )

            # Init Model
            model = BirdModel(model_name=arch, num_classes=num_classes, pretrained=True)
            model.to(device)

            # Optimizer
            optimizer = optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )

            # Train
            save_path = os.path.join(WORKING_DIR, f"model_{arch}_fold_{fold}.pth")
            model, best_auc = train_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                device,
                num_epochs=NUM_EPOCHS,
                patience=PATIENCE,
                save_path=save_path,
            )

            trained_models_fold[arch] = model

            del optimizer
            torch.cuda.empty_cache()

        # Evaluate Ensemble on this fold's validation set (OOF)
        print(f"Generating OOF predictions for Fold {fold}...")
        fold_probs = evaluate_ensemble_on_val(df_val_fold, trained_models_fold, device)

        # Store predictions
        for i, rid in enumerate(df_val_fold["rec_id"].values):
            global_idx = id_to_idx[rid]
            oof_preds[global_idx] = fold_probs[i]
            oof_targets[global_idx] = df_val_fold.iloc[i][label_cols].values

        # Clean up fold models
        del trained_models_fold
        torch.cuda.empty_cache()

    # 4. Global Metrics
    print("\nCalculating Global OOF Metric...")
    final_metric = calculate_metrics(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Load images for analysis
    image_dict = load_and_cache_images(df_full, WORKING_DIR, load_cached_data=True)
    perform_failure_analysis(df_full, oof_targets, oof_preds, image_dict)

    # 6. Submission
    if final_metric > THRESHOLD:
        print(
            f"\nMetric {final_metric} > Threshold {THRESHOLD}. Generating submission..."
        )

        # Free memory before inference
        del image_dict
        torch.cuda.empty_cache()

        # Run inference using the library function
        run_inference(
            models_dir=WORKING_DIR,
            output_dir="./submission",
            batch_size=BATCH_SIZE,
            num_workers=2,
            load_cached_data=True,
        )
    else:
        print(f"\nMetric {final_metric} <= Threshold {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
