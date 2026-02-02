import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders
from library.models import get_model
from library.engine import run_two_phase_training
from library.inference import predict_with_tta, save_submission


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger(name="main_runner")

    # Override Config for Fast Baseline Execution
    # Reducing epochs to ensure completion within strict time limits while demonstrating the idea
    Config.EPOCHS_HEAD = 1
    Config.EPOCHS_FINE = 4

    logger.info("Starting Heterogeneous Ensemble Baseline Run")
    logger.info(
        f"Configuration: Head Epochs={Config.EPOCHS_HEAD}, Fine Epochs={Config.EPOCHS_FINE}"
    )

    # 2. Data Loading
    # Load cached data if available, otherwise process
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Load raw validation metadata for failure analysis later
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # 3. Training Loop (Heterogeneous Ensemble)
    # We will train one model per architecture on the provided train/val split
    trained_models = []
    val_preds_accumulator = []
    test_preds_accumulator = []

    for arch_name in Config.MODEL_ARCHS:
        logger.info(f"\n{'='*40}\nProcessing Architecture: {arch_name}\n{'='*40}")

        # Initialize Model
        model = get_model(arch_name, num_classes=len(class_names), pretrained=True)

        # Define Checkpoint Path
        # Sanitize model name for filename
        safe_arch_name = arch_name.replace(".", "_")
        checkpoint_path = os.path.join(Config.WORKING_DIR, f"{safe_arch_name}_best.pth")

        # Train Model
        model = run_two_phase_training(model, train_loader, val_loader, checkpoint_path)
        trained_models.append(model)

        # --- Inference for Ensemble ---
        logger.info(f"Generating predictions for {arch_name}...")

        # Validation Inference
        # Note: val_loader returns (image, label), so predict_with_tta returns (labels, probs)
        # We ignore the returned labels/ids here and rely on loader order (shuffle=False)
        _, val_probs = predict_with_tta(model, val_loader)
        val_preds_accumulator.append(val_probs)

        # Test Inference
        # test_loader returns (image, id), so predict_with_tta returns (ids, probs)
        test_ids, test_probs = predict_with_tta(model, test_loader)
        test_preds_accumulator.append(test_probs)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # 4. Ensemble Aggregation
    logger.info("\nAggregating Ensemble Predictions...")

    # Average probabilities (Soft Voting)
    ensemble_val_probs = np.mean(val_preds_accumulator, axis=0)
    ensemble_test_probs = np.mean(test_preds_accumulator, axis=0)

    # 5. Validation & Metric
    # Get Ground Truth Labels
    # val_loader.dataset.df contains the 'label_idx' column
    y_true = val_loader.dataset.df["label_idx"].values

    # Calculate Log Loss
    # Clip probabilities to avoid log(0) error, though log_loss handles it usually
    final_metric = log_loss(
        y_true, ensemble_val_probs, labels=list(range(len(class_names)))
    )

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    logger.info("\nPerforming Failure Analysis...")

    # Calculate per-sample Cross Entropy Loss
    # CE = -log(p_true)
    # Gather probability assigned to the true class for each sample
    # Using numpy advanced indexing
    rows = np.arange(len(y_true))
    true_class_probs = ensemble_val_probs[rows, y_true]

    # Clip to epsilon to avoid inf
    epsilon = 1e-15
    true_class_probs = np.clip(true_class_probs, epsilon, 1.0)
    sample_losses = -np.log(true_class_probs)

    # Merge with metadata features
    # We assume val_meta_df order matches val_loader order (both read from same CSV, no shuffle in loader)
    # We need to compute image features: width, height, aspect ratio, area
    # The metadata file has 'file_path', we need to read images or if we trust the analysis script...
    # The analysis script isn't available to import, so we must re-calculate or infer.
    # However, reading 1800 images might be slow.
    # Let's use the provided 'val.csv' and assume we need to read image dims.
    # To be fast, we'll read a few or just read all. 1800 images is fast to read dims.

    widths = []
    heights = []

    # Construct full paths
    full_paths = [os.path.join(Config.INPUT_DIR, p) for p in val_meta_df["file_path"]]

    # Quick dimension check using PIL (lazy loading)
    from PIL import Image

    for p in full_paths:
        try:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except:
            widths.append(0)
            heights.append(0)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.divide(
        widths, heights, out=np.zeros_like(widths, dtype=float), where=heights != 0
    )
    areas = widths * heights

    # Calculate Correlations
    features = {
        "Width": widths,
        "Height": heights,
        "Aspect Ratio": aspect_ratios,
        "Area": areas,
    }

    print("Correlation between Error (Log Loss) and Input Features:")
    for name, feature_vals in features.items():
        if len(feature_vals) != len(sample_losses):
            logger.warning(f"Length mismatch for {name}. Skipping.")
            continue

        corr, _ = pearsonr(sample_losses, feature_vals)
        print(f"{name}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.14144190501755333

    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric {final_metric} meets threshold {THRESHOLD}. Generating submission."
        )
        save_submission(test_ids, ensemble_test_probs, class_names)
    else:
        logger.warning(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped (or saved anyway if required by pipeline logic, but following prompt strictness)."
        )
        # The prompt says "Generate predictions... If and only if...".
        # However, usually pipelines expect a file. I will follow the prompt strictly.
        pass


if __name__ == "__main__":
    main()
