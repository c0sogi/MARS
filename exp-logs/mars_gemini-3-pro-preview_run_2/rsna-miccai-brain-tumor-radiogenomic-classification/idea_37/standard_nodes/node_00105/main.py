import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet
from library.train_eval import run_training, generate_submission

# Initialize Logger
logger = get_logger("Runfile")


def get_validation_predictions(model, loader, device):
    """
    Runs inference on the validation set to retrieve raw targets and probabilities
    for failure analysis.
    """
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # Apply sigmoid to logits to get probabilities
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy().flatten())

    return np.array(all_targets), np.array(all_preds)


def perform_failure_analysis(val_df, targets, preds):
    """
    Analyzes the correlation between prediction error and input data characteristics
    (e.g., slice counts, file sizes) to identify systematic failures.
    """
    logger.info("Starting Failure Analysis on Validation Set...")

    # Calculate absolute error
    errors = np.abs(targets - preds)
    val_df = val_df.copy()
    val_df["error"] = errors
    val_df["target"] = targets
    val_df["prediction"] = preds

    # Extract metadata features for correlation
    # We look at slice counts and average file sizes as proxies for scan quality/resolution
    feature_stats = []

    for idx, row in val_df.iterrows():
        stats = {}
        for mod in Config.MODALITIES:
            # Construct path
            mod_path = os.path.join(Config.INPUT_DIR, row[f"path_{mod}"])
            if os.path.exists(mod_path):
                files = os.listdir(mod_path)
                stats[f"{mod}_count"] = len(files)
                # Calculate average file size (proxy for resolution/info content)
                if len(files) > 0:
                    total_size = sum(
                        os.path.getsize(os.path.join(mod_path, f)) for f in files
                    )
                    stats[f"{mod}_avg_size"] = total_size / len(files)
                else:
                    stats[f"{mod}_avg_size"] = 0
            else:
                stats[f"{mod}_count"] = 0
                stats[f"{mod}_avg_size"] = 0
        feature_stats.append(stats)

    stats_df = pd.DataFrame(feature_stats)
    analysis_df = pd.concat([val_df.reset_index(drop=True), stats_df], axis=1)

    # Calculate correlations with error
    logger.info("Correlation between Error Magnitude and Metadata Features:")
    numeric_cols = [c for c in stats_df.columns]

    correlations = {}
    for col in numeric_cols:
        if analysis_df[col].std() > 0:  # Avoid constant columns
            corr, _ = pearsonr(analysis_df["error"], analysis_df[col])
            correlations[col] = corr
            print(f"Feature: {col:20s} | Correlation with Error: {corr:.4f}")

    # Identify highest correlation
    if correlations:
        max_feat = max(correlations, key=lambda k: abs(correlations[k]))
        logger.info(
            f"Strongest predictor of error: {max_feat} (r={correlations[max_feat]:.4f})"
        )


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution started. Device: {device}")

    # 2. Train the Model
    # We use the full dataset (debug_limit=None) but rely on the small dataset size
    # and limited epochs (20) in Config to ensure fast execution.
    logger.info("Initiating Training Pipeline...")
    run_training(load_cached_data=True, debug_limit=None)

    # 3. Load Best Model for Validation & Analysis
    logger.info("Loading best model for validation assessment...")
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError("Best model not found. Training may have failed.")

    model = AsymmetricEfficientNet()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)

    # Get Validation Data
    _, val_loader, _ = get_dataloaders(load_cached_data=True, debug_limit=None)

    # 4. Validation Inference
    targets, preds = get_validation_predictions(model, val_loader, device)

    # Compute Metric
    final_auc = roc_auc_score(targets, preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    # Load valid IDs to align metadata with predictions
    # Cite debug_lesson_4: Validate Cache Schema Before Consumption
    if os.path.exists(Config.CACHE_VAL_IDS):
        valid_ids = np.load(Config.CACHE_VAL_IDS)
        # Filter and reorder metadata to match the cached data order
        val_metadata = val_metadata.set_index("BraTS21ID").loc[valid_ids].reset_index()

    # Ensure alignment (dataloader should preserve order if shuffle=False, which it is)
    if len(val_metadata) == len(targets):
        perform_failure_analysis(val_metadata, targets, preds)
    else:
        logger.warning(
            f"Validation metadata length mismatch (Metadata: {len(val_metadata)}, Preds: {len(targets)}). Skipping detailed failure analysis."
        )

    # 6. Conditional Submission
    submission_threshold = 0.6321818181818182

    if final_auc > submission_threshold:
        logger.info(
            f"Validation AUC ({final_auc:.6f}) exceeds threshold ({submission_threshold:.6f}). Generating submission..."
        )
        generate_submission(load_cached_data=True, debug_limit=None)
    else:
        logger.info(
            f"Validation AUC ({final_auc:.6f}) did not exceed threshold ({submission_threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
