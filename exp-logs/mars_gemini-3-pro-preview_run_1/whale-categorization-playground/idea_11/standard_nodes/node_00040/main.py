import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
import time

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import fit_model
from library.inference import predict_ensemble
from library.dataset import get_dataloaders
from library.model import WhaleDenseNet
from library.utils import seed_everything, calculate_map5


def run_ensemble_validation(checkpoint_paths, val_loader, device, idx_to_class):
    """
    Evaluates the ensemble of models on the validation set.
    Applies TTA (Horizontal Flip) and averages logits across all models.
    Returns the MAP@5 score, predictions, and targets.
    """
    print("Starting Ensemble Validation...")
    models = []

    # Load all trained ensemble members
    for ckpt_path in checkpoint_paths:
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found at {ckpt_path}")
            continue

        try:
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

            # Initialize model
            model = WhaleDenseNet(num_classes=len(idx_to_class), pretrained=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")

    if not models:
        print("Error: No models loaded for validation.")
        return 0.0, [], []

    all_preds = []
    all_targets = []

    # Inference Loop
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            ensemble_logits = None

            for model in models:
                # 1. Original Forward Pass
                logits_orig = model(images, labels=None)

                # 2. TTA Forward Pass (Horizontal Flip)
                if Config.TTA_FLIP:
                    images_flip = torch.flip(images, dims=[3])
                    logits_flip = model(images_flip, labels=None)
                    # Average logits for this model
                    logits_model = (logits_orig + logits_flip) / 2.0
                else:
                    logits_model = logits_orig

                # Accumulate ensemble logits
                if ensemble_logits is None:
                    ensemble_logits = logits_model
                else:
                    ensemble_logits += logits_model

            # Average across ensemble members
            ensemble_logits /= len(models)

            # Get Top 5 Predictions
            _, top_indices = torch.topk(ensemble_logits, k=5, dim=1)

            all_preds.extend(top_indices.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    # Calculate MAP@5
    map5 = calculate_map5(all_preds, all_targets)

    return map5, all_preds, all_targets


def analyze_failures(val_csv_path, preds, targets):
    """
    Performs failure analysis by correlating error magnitude with input features.
    """
    print("Performing Failure Analysis...")

    # Load validation metadata
    df_val = pd.read_csv(val_csv_path)

    # Align dataframe with predictions (assuming strict ordering from DataLoader with shuffle=False)
    if len(df_val) != len(preds):
        print(
            f"Warning: Validation DataFrame length ({len(df_val)}) does not match predictions ({len(preds)}). Truncating to match."
        )
        df_val = df_val.iloc[: len(preds)].copy()

    # Calculate Average Precision (AP) per sample
    # AP is 1/rank if correct label is in top 5, else 0
    aps = []
    for p, t in zip(preds, targets):
        ap = 0.0
        for rank, pred_idx in enumerate(p):
            if pred_idx == t:
                ap = 1.0 / (rank + 1)
                break
        aps.append(ap)

    df_val["AP"] = aps
    df_val["Error"] = 1.0 - df_val["AP"]

    # Extract image features (Width, Height, Intensity)
    widths = []
    heights = []
    intensities = []

    for _, row in df_val.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                # Normalize intensity to 0-1
                intensities.append(np.mean(img) / 255.0)
            else:
                widths.append(0)
                heights.append(0)
                intensities.append(0)
        else:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    df_val["Width"] = widths
    df_val["Height"] = heights
    # Add small epsilon to avoid division by zero
    df_val["AspectRatio"] = np.array(widths) / (np.array(heights) + 1e-6)
    df_val["Intensity"] = intensities

    # Calculate Correlations
    correlations = {}
    features = ["Width", "Height", "AspectRatio", "Intensity"]

    print("Correlation between Error Magnitude (1-AP) and Input Features:")
    for feat in features:
        if df_val[feat].std() > 0:
            corr = df_val["Error"].corr(df_val[feat])
            correlations[feat] = corr
        else:
            correlations[feat] = 0.0
        print(f"  {feat}: {correlations[feat]:.16f}")


def main():
    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 1. Train Ensemble Members
    # -------------------------------------------------------------------------
    checkpoint_paths = []

    for seed in Config.ENSEMBLE_SEEDS:
        print(f"\n{'='*40}")
        print(f"Training Ensemble Member with Seed: {seed}")
        print(f"{'='*40}")

        # Train model
        fit_model(seed)

        # Store path to best model
        ckpt_path = os.path.join(
            Config.WORKING_DIR, f"seed_{seed}", "model_best.pth.tar"
        )
        checkpoint_paths.append(ckpt_path)

    # -------------------------------------------------------------------------
    # 2. Ensemble Validation
    # -------------------------------------------------------------------------
    print(f"\n{'='*40}")
    print("Evaluating Ensemble on Validation Set")
    print(f"{'='*40}")

    # Load DataLoaders (reusing cached class mapping)
    _, val_loader, test_loader, class_to_idx, idx_to_class = get_dataloaders(
        load_cached_data=True, verbose=False
    )

    # Run validation
    val_map5, val_preds, val_targets = run_ensemble_validation(
        checkpoint_paths, val_loader, device, idx_to_class
    )

    # Print Metric (Required Format)
    print(f"Final Validation Metric: {val_map5}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    analyze_failures(Config.VAL_CSV, val_preds, val_targets)

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    threshold = 0.6545824094604581

    if val_map5 > threshold:
        print(f"\nValidation score ({val_map5}) exceeds threshold ({threshold}).")
        print("Generating submission for Test Set...")
        predict_ensemble(checkpoint_paths, test_loader, device)
    else:
        print(
            f"\nValidation score ({val_map5}) does not exceed threshold ({threshold})."
        )
        print("Submission skipped.")


if __name__ == "__main__":
    main()
