import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import setup_logger, seed_everything
from library.training import run_training
from library.inference import generate_submission
from library.feature_extraction import extract_dataset
from library.densification import prepare_inference_data

# Initialize logger
logger = setup_logger("runfile")


def evaluate_holdout():
    """
    Evaluates the trained ensemble on the hold-out validation set (metadata/val.csv).
    Returns the final log loss metric, per-sample losses, and original tabular features.
    """
    logger.info("Starting evaluation on hold-out validation set...")

    # 1. Load Validation Data
    # extract_dataset handles caching
    img_feats, tab_feats, ids, labels = extract_dataset(
        split="val", load_cached_data=True
    )

    # 2. Prepare Inference Data (Canonical Centroids 3x)
    # Returns X_img: (N*3, D), X_tab: (N*3, T), ids_expanded: (N*3,), y_expanded: (N*3,)
    # We use this for prediction
    X_img_val, X_tab_val, ids_expanded, y_expanded = prepare_inference_data(
        img_features=img_feats,
        tab_features=tab_feats,
        ids=ids,
        labels=labels,
        cache_suffix="val_holdout",
        load_cached_data=True,
    )

    # Concatenate for pipeline
    X_val = np.hstack([X_img_val, X_tab_val])

    # 3. Load Models and Classes
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    classes_path = os.path.join(models_dir, "classes.pkl")

    if not os.path.exists(classes_path):
        logger.error("Classes file not found. Cannot evaluate.")
        return None, None, None

    classes = joblib.load(classes_path)

    model_files = [
        f
        for f in os.listdir(models_dir)
        if f.startswith("pipeline_fold_") and f.endswith(".pkl")
    ]
    if not model_files:
        logger.error("No models found.")
        return None, None, None

    # 4. Ensemble Prediction
    n_samples = len(ids)
    n_classes = len(classes)
    avg_probs = np.zeros((n_samples, n_classes))

    for model_file in model_files:
        path = os.path.join(models_dir, model_file)
        pipeline = joblib.load(path)

        # Predict on expanded set (N*3)
        probs_expanded = pipeline.predict_proba(X_val)

        # Aggregate Centroids: (N*3, C) -> (N, 3, C) -> (N, C)
        probs_reshaped = probs_expanded.reshape(n_samples, 3, n_classes)
        probs_mean = probs_reshaped.mean(axis=1)

        avg_probs += probs_mean

    avg_probs /= len(model_files)

    # 5. Calculate Metric
    # Clip probabilities to avoid log(0) and normalize
    avg_probs_clipped = np.clip(avg_probs, 1e-15, 1 - 1e-15)
    avg_probs_clipped = avg_probs_clipped / avg_probs_clipped.sum(axis=1, keepdims=True)

    final_metric = log_loss(labels, avg_probs_clipped, labels=classes)

    # 6. Calculate per-sample loss for failure analysis
    # Map string labels to indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[l] for l in labels])

    # Select probability of true class
    prob_true = avg_probs_clipped[np.arange(n_samples), y_indices]
    sample_losses = -np.log(prob_true)

    return final_metric, sample_losses, tab_feats


def analyze_failures(sample_losses, tab_features):
    """
    Performs failure analysis by correlating sample losses with tabular features.
    """
    logger.info("Performing failure analysis...")

    if sample_losses is None or tab_features is None:
        logger.warning("Skipping failure analysis due to missing data.")
        return

    # Calculate correlation between each feature and the loss
    n_features = tab_features.shape[1]
    correlations = []

    # tab_features is (N, 192)
    for i in range(n_features):
        feat_vals = tab_features[:, i]
        # Check for constant features to avoid warnings
        if np.std(feat_vals) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(feat_vals, sample_losses)
        correlations.append(corr)

    correlations = np.array(correlations)

    # Get top 5 positive correlations (features associated with high error)
    top_indices = np.argsort(correlations)[::-1][:5]

    logger.info("Top 5 features correlated with high error:")
    for idx in top_indices:
        logger.info(f"Feature index {idx}: Correlation = {correlations[idx]:.4f}")


def main():
    seed_everything(Config.SEED)

    # 1. Train the model (K-Fold Ensemble)
    logger.info("Step 1: Training Ensemble...")
    run_training(debug=False, load_cache=True)

    # 2. Validate on Hold-out Set
    logger.info("Step 2: Evaluating on Hold-out Validation Set...")
    final_metric, sample_losses, tab_feats = evaluate_holdout()

    if final_metric is not None:
        # Print exact string as requested
        print(f"Final Validation Metric: {final_metric}")
    else:
        print("Final Validation Metric: NaN")

    # 3. Failure Analysis
    if sample_losses is not None:
        analyze_failures(sample_losses, tab_feats)

    # 4. Generate Submission
    logger.info("Step 4: Generating Submission...")
    generate_submission(load_cached_features=True, load_cached_inference=True)


if __name__ == "__main__":
    main()
