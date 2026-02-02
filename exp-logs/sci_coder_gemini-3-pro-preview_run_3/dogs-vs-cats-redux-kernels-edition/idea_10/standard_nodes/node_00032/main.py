import os
import gc
import cv2
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything, get_logger, calculate_log_loss
from library.data_loader import get_dataloaders
from library.model_factory import create_model
from library.trainer import run_swa_training
from library.inference import run_inference, predict_model

# --- Configuration Overrides for Fast Baseline ---
# Adjusting epochs to ensure execution within time limits while maintaining SWA logic.
Config.NUM_EPOCHS = 5
Config.SWA_START_EPOCH = 3
Config.USE_SWA = True


def main():
    # Initialize Logger
    logger = get_logger("runfile")
    logger.info("Starting execution of runfile.py")

    # Set Seeds
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Training Phase
    # -------------------------------------------------------------------------
    logger.info("Starting Training Phase...")

    # Iterate through each model in the heterogeneous ensemble
    for model_name in Config.MODEL_SPECS.keys():
        logger.info(f"--- Training Model: {model_name} ---")

        # Get DataLoaders
        # We use the full dataset but fewer epochs for speed
        train_loader, val_loader, _ = get_dataloaders(model_name, load_cached_data=True)

        # Create Model
        # Load pretrained weights for training
        model = create_model(model_name, pretrained=True)

        # Run Training (handles SWA and saving weights)
        run_swa_training(model, train_loader, val_loader, model_name)

        # Cleanup to free GPU memory
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Training Phase Complete.")

    # -------------------------------------------------------------------------
    # 2. Ensemble Validation Phase
    # -------------------------------------------------------------------------
    logger.info("Starting Ensemble Validation Phase...")

    # Load validation ground truth
    val_df = pd.read_csv(Config.VAL_METADATA)
    y_true = val_df["label"].values

    model_preds = []

    # Generate predictions for each model on the validation set
    for model_name in Config.MODEL_SPECS.keys():
        logger.info(f"Validating model: {model_name}")

        # Re-initialize loader and model
        _, val_loader, _ = get_dataloaders(model_name, load_cached_data=True)
        model = create_model(model_name, pretrained=False)

        # Determine which weights to load (SWA preferred)
        swa_path = os.path.join(Config.WORKING_DIR, f"{model_name}_swa.pth")
        best_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

        if Config.USE_SWA and os.path.exists(swa_path):
            checkpoint_path = swa_path
        elif os.path.exists(best_path):
            checkpoint_path = best_path
        else:
            logger.warning(f"No weights found for {model_name}, skipping in ensemble.")
            continue

        # Load weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)

        # Generate probabilities (using TTA to match inference capability)
        # val_loader returns (image, label), predict_model handles the tuple unpacking
        preds = predict_model(model, val_loader, Config.DEVICE, use_tta=Config.TTA_FLIP)
        model_preds.append(preds)

        # Cleanup
        del model, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    if not model_preds:
        logger.error("No models were successfully validated. Exiting.")
        return

    # Aggregate Predictions (Arithmetic Mean)
    ensemble_preds = np.mean(model_preds, axis=0)

    # Calculate Metric
    final_metric = calculate_log_loss(y_true, ensemble_preds)

    # PRINT FINAL METRIC (Required Format)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(y_true - ensemble_preds)

    # Extract metadata features from images
    # We need to read the files to get width, height, file_size
    widths = []
    heights = []
    file_sizes = []

    # Iterate through validation files
    # Note: This adds some time, but is required for the analysis
    for filepath in val_df["filepath"]:
        full_path = os.path.join(Config.INPUT_DIR, filepath)

        if os.path.exists(full_path):
            # File Size
            file_sizes.append(os.path.getsize(full_path))

            # Dimensions (OpenCV reads as H, W, C)
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        else:
            # Fallback for missing files (should not happen based on metadata checks)
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "file_size": file_sizes}
    )

    # Calculate Correlations
    print("Failure Analysis (Correlation with Error Magnitude):")
    for feature in ["width", "height", "file_size"]:
        # Handle cases with constant values or NaNs to avoid crashes
        if analysis_df[feature].std() > 0:
            corr, _ = pearsonr(analysis_df[feature], analysis_df["error"])
            print(f"{feature}: {corr:.10f}")
        else:
            print(f"{feature}: NaN (No variance)")

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold defined in task description
    SUBMISSION_THRESHOLD = 0.009241249605204765

    if final_metric < SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation metric {final_metric} < {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        run_inference()
    else:
        logger.info(
            f"Validation metric {final_metric} >= {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
