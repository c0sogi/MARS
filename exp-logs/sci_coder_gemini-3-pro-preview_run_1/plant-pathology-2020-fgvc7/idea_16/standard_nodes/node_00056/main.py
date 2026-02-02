import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_device, get_logger, load_model
from library.data import AppleDataset, get_transforms
from library.model import AppleResNet34
from library.production import run_production_phase
from library.engine import predict

# Initialize Logger
logger = get_logger(name="main")


def load_validation_data():
    """Loads the hold-out validation dataset based on metadata."""
    logger.info(f"Loading validation metadata from {Config.VAL_METADATA_PATH}")
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # We use mode='test' to get (image, image_id) compatible with engine.predict
    # We will retrieve labels from the dataframe directly for metric calculation
    val_dataset = AppleDataset(
        df_val, transforms=get_transforms(data="valid"), mode="test"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return df_val, val_loader


def ensemble_predict(model_paths, loader, device):
    """Generates averaged predictions from multiple models."""
    logger.info(f"Starting ensemble inference with {len(model_paths)} models...")

    avg_preds = None

    for i, path in enumerate(model_paths):
        logger.info(f"Loading model {i+1}/{len(model_paths)}: {path}")

        # Initialize and load model
        model = AppleResNet34(pretrained=False)  # Weights loaded from checkpoint
        model = load_model(model, path, device)
        model.to(device)
        model.eval()

        # Predict
        ids, preds = predict(model, loader, device)

        if avg_preds is None:
            avg_preds = preds
        else:
            avg_preds += preds

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Average predictions
    avg_preds /= len(model_paths)

    return ids, avg_preds


def perform_failure_analysis(df, preds, targets):
    """
    Analyzes correlation between error magnitude and image meta-features.
    """
    logger.info("Performing Failure Analysis...")

    # Calculate Error Magnitude
    # Error = 1.0 - probability assigned to the correct class
    # We assume targets are one-hot or we take argmax
    true_indices = np.argmax(targets, axis=1)

    # Extract probability of the true class for each sample
    # preds is [N, 4], true_indices is [N]
    prob_correct = preds[np.arange(len(preds)), true_indices]
    errors = 1.0 - prob_correct

    # Extract Meta-Features
    widths = []
    heights = []
    aspect_ratios = []
    intensities = []

    logger.info("Extracting image features for correlation analysis...")
    for idx, row in df.iterrows():
        # Construct path relative to input
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for missing images (should not happen based on checks)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            intensities.append(0)
            continue

        h, w, c = img.shape
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)

        # Mean intensity (normalize to 0-1)
        intensities.append(img.mean() / 255.0)

    # Calculate Correlations
    features = {
        "Width": widths,
        "Height": heights,
        "Aspect Ratio": aspect_ratios,
        "Mean Intensity": intensities,
    }

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    for name, values in features.items():
        if len(values) != len(errors):
            logger.warning(f"Skipping {name} due to length mismatch.")
            continue

        # Pearson correlation
        corr, _ = pearsonr(errors, values)
        print(f"  {name}: {corr:.4f}")
    print("")


def main():
    # 1. Setup
    seed_everything(Config.SEEDS[0])
    device = get_device()
    logger.info(f"Device: {device}")

    # 2. Load Validation Data (Used for Early Stopping and Final Assessment)
    logger.info("Loading Validation Data...")
    df_val, val_loader = load_validation_data()

    # 3. Production Phase
    # Train ensemble with Early Stopping on Validation Set
    # Cite solution_lesson_node_00055: Seed Averaging Ensembles.
    logger.info("=== Stage 2: Production (Seed Ensemble) ===")
    model_paths = run_production_phase(val_loader=val_loader, load_cached_data=True)

    # 4. Validation Assessment
    logger.info("=== Stage 3: Validation Assessment ===")

    # Generate Ensemble Predictions on Validation Set
    _, val_preds = ensemble_predict(model_paths, val_loader, device)

    # Get Targets
    val_targets = df_val[Config.TARGET_COLS].values

    # Calculate Metric
    # Mean column-wise ROC AUC
    try:
        val_auc = roc_auc_score(
            val_targets, val_preds, average="macro", multi_class="ovr"
        )
    except Exception as e:
        logger.error(f"Error calculating AUC: {e}")
        val_auc = 0.0

    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis
    perform_failure_analysis(df_val, val_preds, val_targets)

    # 5. Submission
    # Threshold check
    THRESHOLD = 0.9901680711448418

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        test_dataset = AppleDataset(
            df_test, transforms=get_transforms(data="valid"), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Ensemble Predict
        test_ids, test_preds = ensemble_predict(model_paths, test_loader, device)

        # Create Submission DataFrame
        submission = pd.DataFrame({"image_id": test_ids})
        for i, col in enumerate(Config.TARGET_COLS):
            submission[col] = test_preds[:, i]

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Validation metric ({val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
