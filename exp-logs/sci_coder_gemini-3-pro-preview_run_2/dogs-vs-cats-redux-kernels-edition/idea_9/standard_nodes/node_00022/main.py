import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.train import run_training
from library.inference import predict_test_set
from library.dataset import PetDataset, get_transforms
from library.models import get_model
from library.utils import load_checkpoint, seed_everything


def main():
    # 1. Configuration Override for Fast Baseline
    # We need to ensure the code completes within the 2-hour limit.
    # We have 10 models to train (2 archs * 5 folds).
    # On an A100 GPU, we can increase batch size to 64 and limit epochs to 4
    # to ensure the entire pipeline (training + validation + inference) finishes on time.
    print("Configuring for fast baseline execution...")
    Config.EPOCHS = 4
    Config.BATCH_SIZE = 64

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Run Training
    print("Starting training pipeline...")
    # load_cached_data=False ensures we create folds from scratch using the provided metadata
    run_training(load_cached_data=False)
    print("Training pipeline completed.")

    # 3. Validation and Failure Analysis
    print("Starting validation and failure analysis...")

    # Load metadata
    val_meta_df = pd.read_csv(Config.VAL_META)
    folds_df = pd.read_parquet(os.path.join(Config.WORKING_DIR, "folds.parquet"))

    # Merge to get fold assignments for the hold-out validation set
    # We merge on 'filepath' to map the validation samples to the folds they were assigned to
    val_merged = pd.merge(
        val_meta_df, folds_df[["filepath", "fold"]], on="filepath", how="left"
    )

    if val_merged["fold"].isnull().any():
        print(
            "Warning: Some validation samples were not found in folds.parquet. Dropping them."
        )
        val_merged = val_merged.dropna(subset=["fold"])

    val_merged["fold"] = val_merged["fold"].astype(int)

    device = Config.DEVICE

    # Container for OOF predictions
    val_merged["oof_prob"] = np.nan

    # Iterate through each fold to perform OOF validation
    for fold in range(Config.N_FOLDS):
        print(f"Validating Fold {fold}...")

        # Get validation samples that belong to this fold
        fold_val_df = val_merged[val_merged["fold"] == fold].reset_index(drop=True)

        if len(fold_val_df) == 0:
            continue

        # Create Dataset/Loader
        # Note: We use 'valid' transforms which are deterministic (Resize + Normalize)
        ds = PetDataset(
            fold_val_df, transforms=get_transforms(data_type="valid"), mode="train"
        )
        dl = DataLoader(
            ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Accumulator for ensemble predictions for this fold
        fold_preds = np.zeros(len(fold_val_df))

        for model_name in Config.MODEL_ARCHS:
            # Load the model trained for this fold (which used the other folds for training)
            ckpt_path = os.path.join(
                Config.CHECKPOINT_DIR, f"best_{model_name}_fold_{fold}.pth"
            )

            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
                continue

            model = get_model(model_name, pretrained=False, num_classes=1)
            model = model.to(device)
            load_checkpoint(ckpt_path, model, device=device)

            model.eval()
            preds = []
            with torch.no_grad():
                for images, _ in dl:
                    images = images.to(device)

                    # Forward pass
                    logits = model(images)
                    probs = torch.sigmoid(logits).view(-1)

                    # Apply Test-Time Augmentation (Horizontal Flip)
                    if Config.TTA_FLIP:
                        images_flipped = torch.flip(images, dims=[3])
                        logits_flipped = model(images_flipped)
                        probs_flipped = torch.sigmoid(logits_flipped).view(-1)
                        probs = (probs + probs_flipped) / 2.0

                    preds.extend(probs.cpu().numpy())

            fold_preds += np.array(preds)

            # Clean up
            del model
            torch.cuda.empty_cache()

        # Average predictions over the architectures (Heterogeneous Ensemble)
        fold_preds /= len(Config.MODEL_ARCHS)

        # Store predictions back to the dataframe
        val_merged.loc[val_merged["fold"] == fold, "oof_prob"] = fold_preds

    # Calculate Final Metric
    valid_results = val_merged.dropna(subset=["oof_prob"])

    y_true = valid_results["label"].values
    y_pred = valid_results["oof_prob"].values

    final_metric = log_loss(y_true, y_pred)
    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    valid_results = valid_results.copy()
    valid_results["error"] = np.abs(valid_results["label"] - valid_results["oof_prob"])

    # Calculate image metadata for correlation analysis
    widths = []
    heights = []
    aspect_ratios = []

    print("Extracting metadata features from validation images...")
    for filepath in valid_results["filepath"]:
        full_path = os.path.join(Config.INPUT_DIR, filepath)
        img = cv2.imread(full_path)
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
        else:
            widths.append(np.nan)
            heights.append(np.nan)
            aspect_ratios.append(np.nan)

    valid_results["width"] = widths
    valid_results["height"] = heights
    valid_results["aspect_ratio"] = aspect_ratios

    # Compute Correlations
    correlations = valid_results[["error", "width", "height", "aspect_ratio"]].corr()[
        "error"
    ]
    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Width: {correlations['width']:.4f}")
    print(f"  Height: {correlations['height']:.4f}")
    print(f"  Aspect Ratio: {correlations['aspect_ratio']:.4f}")

    # 5. Submission
    THRESHOLD = 0.018199009307556684
    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        predict_test_set()
    else:
        print(
            f"\nValidation metric ({final_metric}) did not beat threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
