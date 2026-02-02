import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

import gc
import importlib
import library.config

# Reload config to ensure changes (like BATCH_SIZE) are picked up in persistent runtimes (Cite debug_lesson_1)
importlib.reload(library.config)
from library.config import Config
from library.dataset import DogCatDataset, get_transforms
from library.models import get_model
from library.engine import fit, predict
from library.utils import set_seed, load_checkpoint
from library.inference import predict_ensemble


def analyze_failures(val_df, preds, input_dir):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and image metadata.
    """
    print("\n--- Failure Analysis ---")

    # 1. Calculate Error Magnitude
    # val_df['label'] contains ground truth (0 or 1)
    targets = val_df["label"].values
    # preds are probabilities of class 1
    # Error magnitude: |y - p|
    errors = np.abs(targets - preds)

    # 2. Extract Metadata Features
    # We need to read image dimensions and file sizes
    print(f"Extracting features for {len(val_df)} validation images...")

    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []

    for _, row in val_df.iterrows():
        rel_path = row["filepath"]
        full_path = os.path.join(input_dir, rel_path)

        if os.path.exists(full_path):
            # File Size
            try:
                fsize = os.path.getsize(full_path)
            except OSError:
                fsize = 0
            file_sizes.append(fsize)

            # Dimensions
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
            else:
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # 3. Compute Correlations
    feature_dict = {
        "width": widths,
        "height": heights,
        "aspect_ratio": aspect_ratios,
        "file_size": file_sizes,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, values in feature_dict.items():
        if len(values) != len(errors):
            continue

        # Pearson correlation
        corr, p_val = pearsonr(values, errors)
        print(f"  {name:<15}: Correlation = {corr:.10f} (p-value = {p_val:.10f})")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = DogCatDataset("train", transform=get_transforms("train"))
    val_dataset = DogCatDataset("val", transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # 3. Training Loop
    trained_models = []

    for model_name in Config.MODEL_ARCHS:
        print(f"\nStarting training for architecture: {model_name}")

        # Save Directory
        save_dir = os.path.join(Config.WORKING_DIR, model_name)

        # Check if model is already trained
        if os.path.exists(os.path.join(save_dir, "model_best.pth")):
            print(f"Checkpoint found for {model_name}. Skipping training.")
            trained_models.append(model_name)
            continue

        # Init Model
        model = get_model(model_name, pretrained=True)
        model.to(device)

        # Train
        fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=Config.EPOCHS,
            learning_rate=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            save_dir=save_dir,
        )

        trained_models.append(model_name)

        # Cleanup to free GPU memory for the next model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Ensemble Validation
    print("\nPerforming Ensemble Validation...")

    val_df = pd.read_csv(Config.VAL_CSV)
    ensemble_probs = np.zeros(len(val_df))

    for model_name in trained_models:
        print(f"Evaluating {model_name}...")

        # Load Model
        model = get_model(model_name, pretrained=False)
        model.to(device)
        checkpoint_path = os.path.join(Config.WORKING_DIR, model_name, "model_best.pth")
        load_checkpoint(checkpoint_path, model, device=device)

        # Predict on Val
        # Note: val_loader returns (image, label). predict returns (ids, probs).
        # Here 'ids' will be the labels from the dataset.
        _, probs = predict(model, val_loader, device, use_tta=Config.TTA_FLIP)

        ensemble_probs += np.array(probs)

        # Cleanup to free GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Average
    avg_probs = ensemble_probs / len(trained_models)

    # Calculate Metric
    y_true = val_df["label"].values

    # Numerical stability for Log Loss
    eps = 1e-15
    avg_probs_clipped = np.clip(avg_probs, eps, 1 - eps)

    final_metric = log_loss(y_true, avg_probs_clipped)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    analyze_failures(val_df, avg_probs, Config.INPUT_DIR)

    # 6. Submission Check
    # Threshold defined in task
    THRESHOLD = 0.014050961788691994

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )
        # Pass updated batch size explicitly to avoid stale default arg
        predict_ensemble(batch_size=Config.BATCH_SIZE)
    else:
        print(f"\nValidation metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
