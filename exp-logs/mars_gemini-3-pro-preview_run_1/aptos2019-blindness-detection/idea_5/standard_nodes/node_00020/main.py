import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.data import get_dataloaders
from library.model import RetinopathyModel
from library.utils import seed_everything, compute_score
from library.engine import run_training


def analyze_failures(df_val, preds, true_labels, input_dir):
    """
    Performs failure analysis by correlating error magnitude with image meta-features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error
    errors = np.abs(preds - true_labels)

    # Collect meta-features
    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []
    mean_intensities = []

    print("Extracting meta-features from validation images...")
    for idx, row in df_val.iterrows():
        file_path = os.path.join(input_dir, row["file_path"])

        # File size
        if os.path.exists(file_path):
            file_sizes.append(os.path.getsize(file_path))
        else:
            file_sizes.append(0)

        # Image stats
        img = cv2.imread(file_path)
        if img is not None:
            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            mean_intensities.append(img.mean())
        else:
            # Fallback for missing/corrupt images
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            mean_intensities.append(0)

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "file_size": file_sizes,
            "mean_intensity": mean_intensities,
        }
    )

    # Compute correlations
    features = ["width", "height", "aspect_ratio", "file_size", "mean_intensity"]
    print("\nCorrelation between Error Magnitude and Input Features:")
    for feat in features:
        if analysis_df[feat].std() > 0:
            corr, _ = pearsonr(analysis_df["error"], analysis_df[feat])
            print(f"{feat}: {corr:.6f}")
        else:
            print(f"{feat}: NaN (No variance)")


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Threshold for submission
    SUBMISSION_THRESHOLD = 0.9196387348530078
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # 2. Train Model
    # We use the provided engine function which handles the training loop and saves the best model
    print("Starting Training Phase...")
    run_training()

    # 3. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis Phase...")

    # Load Validation Data
    # Note: We need the raw dataframe for file paths for failure analysis
    df_val = pd.read_csv(Config.val_meta_path)
    _, val_loader, _ = get_dataloaders(batch_size=Config.batch_size)

    # Load Best Model
    model = RetinopathyModel(pretrained=False)
    if not os.path.exists(Config.model_save_path):
        print(f"Error: Model file not found at {Config.model_save_path}")
        return

    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    model.to(device)
    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)

            # Decode Ordinal Predictions
            probs = torch.sigmoid(logits)
            scores = probs.sum(dim=1)
            preds = scores.round().cpu().numpy().astype(int)

            # Decode Targets
            true_labels = targets.sum(dim=1).cpu().numpy().astype(int)

            all_preds.extend(preds)
            all_targets.extend(true_labels)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metric
    qwk = compute_score(all_targets, all_preds)
    print(f"Final Validation Metric: {qwk}")

    # Failure Analysis
    analyze_failures(df_val, all_preds, all_targets, Config.input_dir)

    # 4. Submission
    if qwk > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({qwk}) meets threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Load Test Loader
        _, _, test_loader = get_dataloaders(batch_size=Config.batch_size)

        test_ids = []
        test_preds = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                logits = model(images)
                probs = torch.sigmoid(logits)
                scores = probs.sum(dim=1)
                preds = scores.round().cpu().numpy().astype(int)

                # Clip to valid range
                preds = np.clip(preds, 0, 4)

                test_ids.extend(ids)
                test_preds.extend(preds)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id_code": test_ids, "diagnosis": test_preds})

        # Save
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")
        print(df_sub.head())

    else:
        print(
            f"\nValidation metric ({qwk}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
