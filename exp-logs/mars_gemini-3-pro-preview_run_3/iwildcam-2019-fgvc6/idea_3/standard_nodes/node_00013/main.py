import sys
import os
import pandas as pd
import numpy as np
import torch
import cv2
from scipy.stats import pearsonr
from torch.cuda.amp import autocast

# ------------------------------------------------------------------------------
# 1. Configuration & Setup
# ------------------------------------------------------------------------------
# Import Config and override parameters for this run
from library.config import Config

# Override Config for a fast but effective baseline
# 10 epochs is sufficient for Swin Tiny to converge on this dataset size
# while fitting comfortably within the 2-hour runtime limit on an A100.
Config.NUM_EPOCHS = 10
Config.DEBUG = False

# Import Library Modules
from library.trainer import Trainer
from library.utils import seed_everything, calculate_score, get_logger
from library.inference import generate_submission


def main():
    # Initialize Logger
    logger = get_logger("Runfile")
    logger.info("Starting runfile execution...")

    # Set Seeds
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    logger.info(f"Initializing Trainer (Epochs={Config.NUM_EPOCHS})...")
    trainer = Trainer(debug=Config.DEBUG)

    logger.info("Starting Training Loop...")
    trainer.fit()

    # --------------------------------------------------------------------------
    # 3. Validation & Evaluation
    # --------------------------------------------------------------------------
    logger.info("Starting Validation Evaluation...")

    # Use the best model weights for validation
    if os.path.exists(trainer.checkpoint_path):
        state_dict = torch.load(trainer.checkpoint_path, map_location=Config.DEVICE)
        trainer.model.load_state_dict(state_dict)
        logger.info("Loaded best model checkpoint for validation.")
    else:
        logger.warning("Checkpoint not found. Using current model weights.")

    trainer.model.eval()
    val_loader = trainer.val_loader

    all_preds = []
    all_labels = []

    # Inference loop on validation set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.DEVICE, non_blocking=True)

            with autocast():
                outputs = trainer.model(images)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Metric
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    val_f1 = calculate_score(y_true, y_pred)

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {val_f1}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Get validation metadata to access features
    # val_loader.dataset is AnimalDataset, which has .df attribute
    val_df = val_loader.dataset.df.copy()

    # Ensure alignment
    if len(val_df) != len(y_pred):
        logger.warning(
            f"Shape mismatch: DF {len(val_df)} vs Preds {len(y_pred)}. Truncating to min."
        )
        min_len = min(len(val_df), len(y_pred))
        val_df = val_df.iloc[:min_len]
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

    val_df["Predicted"] = y_pred
    val_df["True"] = y_true
    val_df["Error"] = (val_df["Predicted"] != val_df["True"]).astype(int)

    # Sample a subset for expensive image stats calculation
    SAMPLE_SIZE = 2000
    if len(val_df) > SAMPLE_SIZE:
        analysis_df = val_df.sample(n=SAMPLE_SIZE, random_state=Config.SEED).copy()
    else:
        analysis_df = val_df.copy()

    # Extract Image Features (Width, Height, Mean Intensity, Std Intensity)
    widths = []
    heights = []
    means = []
    stds = []

    logger.info(f"Extracting image stats for {len(analysis_df)} samples...")
    for _, row in analysis_df.iterrows():
        # Path construction matches dataset.py logic
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        try:
            # Read image
            img = cv2.imread(full_path)
            if img is not None:
                h, w, c = img.shape
                widths.append(w)
                heights.append(h)

                # Simple stats (normalize 0-1)
                img_mean = np.mean(img) / 255.0
                img_std = np.std(img) / 255.0
                means.append(img_mean)
                stds.append(img_std)
            else:
                widths.append(np.nan)
                heights.append(np.nan)
                means.append(np.nan)
                stds.append(np.nan)
        except Exception:
            widths.append(np.nan)
            heights.append(np.nan)
            means.append(np.nan)
            stds.append(np.nan)

    analysis_df["Width"] = widths
    analysis_df["Height"] = heights
    analysis_df["PixelMean"] = means
    analysis_df["PixelStd"] = stds

    # Drop failed loads
    analysis_df = analysis_df.dropna()

    # Calculate Correlations
    features_to_check = ["Category", "Width", "Height", "PixelMean", "PixelStd"]

    print("Correlation between Error Magnitude and Input Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Ensure numeric
            if pd.api.types.is_numeric_dtype(analysis_df[feat]):
                # Check for variance to avoid division by zero in correlation
                if analysis_df[feat].std() > 1e-9:
                    corr, _ = pearsonr(analysis_df[feat], analysis_df["Error"])
                    print(f"Correlation Error vs {feat}: {corr}")
                else:
                    print(f"Correlation Error vs {feat}: Undefined (Zero Variance)")
            else:
                print(f"Skipping non-numeric feature: {feat}")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    TARGET_METRIC = 0.9293196996798049

    if val_f1 > TARGET_METRIC:
        logger.info(
            f"Validation Metric ({val_f1}) meets threshold ({TARGET_METRIC}). Generating submission..."
        )
        generate_submission(trainer.checkpoint_path, debug=Config.DEBUG)
    else:
        logger.info(
            f"Validation Metric ({val_f1}) does NOT meet threshold ({TARGET_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    main()
