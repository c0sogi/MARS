import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
import warnings

# Import library modules
from library.utils import seed_everything, get_device, calculate_metrics
from library.dataset import get_loader, load_and_cache_images
from library.models import BirdModel
from library.trainer import train_model
from library.inference import run_inference, TTADataset, predict_with_model, TTA_SHIFTS
from torch.utils.data import DataLoader

# Configuration
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 20  # Fast baseline
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ARCHITECTURES = ["resnet18", "efficientnet_b0", "densenet121"]
WORKING_DIR = "./working/idea_16"
METADATA_DIR = "./metadata"
INPUT_DIR = "./input"
THRESHOLD = 0.0


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

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    # Identify label columns
    label_cols = [c for c in df_train.columns if c.startswith("species_")]
    num_classes = len(label_cols)

    trained_models = {}

    # 3. Train Loop
    for arch in ARCHITECTURES:
        print(f"\nTraining Architecture: {arch}")

        # Get Loaders
        train_loader = get_loader(
            df_train,
            arch,
            phase="train",
            batch_size=BATCH_SIZE,
            num_workers=2,
            load_cached_data=True,
            cache_dir=WORKING_DIR,
        )
        val_loader = get_loader(
            df_val,
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
        save_path = os.path.join(WORKING_DIR, f"model_{arch}_fold_0.pth")
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

        trained_models[arch] = model

        # Clean up
        del optimizer
        torch.cuda.empty_cache()

    # 4. Validation Ensemble & Metrics
    print("\nRunning Validation Ensemble (TTA)...")
    y_pred_val = evaluate_ensemble_on_val(df_val, trained_models, device)
    y_true_val = df_val[label_cols].values

    final_metric = calculate_metrics(y_true_val, y_pred_val)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Load images for analysis
    image_dict = load_and_cache_images(df_val, WORKING_DIR, load_cached_data=True)
    perform_failure_analysis(df_val, y_true_val, y_pred_val, image_dict)

    # 6. Submission
    if final_metric > THRESHOLD:
        print(
            f"\nMetric {final_metric} > Threshold {THRESHOLD}. Generating submission..."
        )

        # Free memory before inference
        del trained_models
        del image_dict
        torch.cuda.empty_cache()

        # Run inference using the library function
        # It expects models in WORKING_DIR with format model_{arch}_fold_{fold}.pth
        # We only have fold 0, which is fine (it will skip others)
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
