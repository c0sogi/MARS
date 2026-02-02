import sys
import os
import pandas as pd
import numpy as np
import torch
import cv2
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.train import run_fold
from library.inference import run_inference
from library.dataset import prepare_folds, get_transforms, PetDataset
from library.models import get_model
from library.utils import seed_everything


def configure_fast_baseline():
    """
    Adjusts configuration for a fast baseline run to meet time constraints.
    """
    print("Configuring for fast baseline execution...")
    # Reduce epochs to ensure completion within 2 hours
    # 10 models * 5 epochs should fit comfortably on A100
    Config.EPOCHS = 5
    print(f"EPOCHS set to {Config.EPOCHS}")


def run_training():
    """
    Trains all models defined in Config.
    """
    print("\n=== Starting Training Phase ===")
    for model_name in Config.MODEL_ARCHS:
        for fold_idx in range(Config.N_FOLDS):
            print(f"\n[Training] Architecture: {model_name} | Fold: {fold_idx}")
            # Check if checkpoint exists to potentially skip, but for baseline we train
            run_fold(model_name, fold_idx)


def generate_oof_predictions():
    """
    Generates Out-Of-Fold predictions for the entire dataset using the trained ensemble.
    Returns a DataFrame with ground truth and aggregated predictions.
    """
    print("\n=== Starting Validation / OOF Generation Phase ===")
    device = torch.device(Config.DEVICE)

    # Load the full dataset with fold assignments
    # This ensures we have the exact same splits as used in training
    df = prepare_folds(load_cached_data=True)

    # Initialize prediction column
    df["pred_prob"] = 0.0

    # Iterate through folds to generate predictions for the validation part of each fold
    for fold_idx in range(Config.N_FOLDS):
        print(f"Generating OOF predictions for Fold {fold_idx}...")

        # Identify validation samples for this fold
        val_mask = df["fold"] == fold_idx
        val_df = df[val_mask].copy()

        if len(val_df) == 0:
            continue

        # Create DataLoader for this validation split
        val_dataset = PetDataset(
            val_df, transforms=get_transforms(mode="val"), mode="val"
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # We will average predictions from all architectures trained on this fold
        fold_ensemble_probs = np.zeros(len(val_df))
        models_loaded = 0

        for model_name in Config.MODEL_ARCHS:
            ckpt_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{model_name}_fold_{fold_idx}.pth"
            )

            if not os.path.exists(ckpt_path):
                print(f"  Warning: Checkpoint {ckpt_path} not found. Skipping model.")
                continue

            # Load Model
            model = get_model(model_name, pretrained=False)
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            model_preds = []

            # Inference Loop
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)

                    # Apply TTA (Horizontal Flip) to match inference strategy
                    logits = model(images)
                    probs = torch.sigmoid(logits)

                    images_flipped = torch.flip(images, dims=[3])
                    logits_flipped = model(images_flipped)
                    probs_flipped = torch.sigmoid(logits_flipped)

                    avg_probs = (probs + probs_flipped) / 2.0
                    model_preds.append(avg_probs.cpu().numpy().flatten())

            models_loaded += 1
            fold_ensemble_probs += np.concatenate(model_preds)

            # Cleanup
            del model
            torch.cuda.empty_cache()

        if models_loaded > 0:
            fold_ensemble_probs /= models_loaded
            # Store predictions in the main dataframe
            df.loc[val_mask, "pred_prob"] = fold_ensemble_probs
        else:
            print(f"  Error: No models found for Fold {fold_idx}. OOF preds will be 0.")

    return df


def perform_failure_analysis(df):
    """
    Analyzes the correlation between prediction error and image metadata.
    """
    print("\n=== Performing Failure Analysis ===")

    # Calculate absolute error
    df["error"] = np.abs(df["label"] - df["pred_prob"])

    print("Extracting metadata features from images...")
    widths = []
    heights = []
    aspect_ratios = []

    # We iterate through the dataframe to read image dimensions
    # This might take a moment but is required for the analysis
    for _, row in df.iterrows():
        filepath = os.path.join(Config.INPUT_DIR, row["filepath"])
        # Use OpenCV to read image dimensions
        img = cv2.imread(filepath)
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
        else:
            # Fallback for missing images (shouldn't happen based on metadata check)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    df["width"] = widths
    df["height"] = heights
    df["aspect_ratio"] = aspect_ratios

    print("\nCorrelations between Error and Metadata:")
    features = ["width", "height", "aspect_ratio"]
    for feat in features:
        if df[feat].std() > 0:
            corr, _ = pearsonr(df["error"], df[feat])
            print(f"Correlation between Error and {feat}: {corr:.10f}")
        else:
            print(f"Correlation between Error and {feat}: N/A (Constant feature)")


def main():
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Adjust config for runtime constraints
    configure_fast_baseline()

    # 1. Train Models
    run_training()

    # 2. Generate OOF Predictions & Validate
    oof_df = generate_oof_predictions()

    # 3. Compute Final Validation Metric
    # Ensure we only calculate on rows that have predictions (in case of partial runs, though we run all)
    # Since we initialize with 0.0, and 0.0 is a valid probability, we assume full coverage.
    y_true = oof_df["label"].values
    y_pred = oof_df["pred_prob"].values

    # Clip predictions to avoid log(0) just in case, though sigmoid handles it
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    final_metric = log_loss(y_true, y_pred, labels=[0, 1])

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    perform_failure_analysis(oof_df)

    # 5. Submission Generation
    THRESHOLD = 0.018199009307556684

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating submission..."
        )
        run_inference()
    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
