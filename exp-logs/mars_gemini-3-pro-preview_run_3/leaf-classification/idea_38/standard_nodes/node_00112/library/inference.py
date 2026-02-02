import os
import joblib
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger, seed_everything
from library.feature_extraction import extract_dataset
from library.densification import prepare_inference_data

# Initialize logger
logger = setup_logger("inference")


def generate_submission(
    load_cached_features: bool = True, load_cached_inference: bool = True
):
    """
    Generates the submission file by aggregating predictions from the trained ensemble.

    Args:
        load_cached_features (bool): Whether to load extracted features from cache.
        load_cached_inference (bool): Whether to load densified inference data from cache.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    logger.info("Starting submission generation...")

    # 1. Load Test Data Features
    # extract_dataset returns raw features:
    # img_feats: (N, 12, D), tab_feats: (N, 192), ids: (N,)
    img_feats, tab_feats, ids, _ = extract_dataset(
        split="test", load_cached_data=load_cached_features
    )

    # 2. Prepare Inference Data (Canonical Centroids 3x)
    # This transforms the 12-view raw features into 3 orthogonal centroids per image.
    # Returns X_img: (N*3, D), X_tab: (N*3, T), ids_expanded: (N*3,)
    X_img_test, X_tab_test, ids_expanded, _ = prepare_inference_data(
        img_features=img_feats,
        tab_features=tab_feats,
        ids=ids,
        labels=None,
        cache_suffix="test",
        load_cached_data=load_cached_inference,
    )

    # Concatenate Visual and Tabular streams for the pipeline
    # Shape: (N*3, 2752)
    X_test = np.hstack([X_img_test, X_tab_test])

    # 3. Load Model Classes
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    classes_path = os.path.join(models_dir, "classes.pkl")

    if not os.path.exists(classes_path):
        raise FileNotFoundError(
            f"Classes file not found at {classes_path}. Please train the model first."
        )

    classes = joblib.load(classes_path)

    # 4. Ensemble Prediction
    n_samples = len(ids)
    n_classes = len(classes)
    avg_probs = np.zeros((n_samples, n_classes))

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found at {models_dir}")

    # Identify all trained fold models
    model_files = [
        f
        for f in os.listdir(models_dir)
        if f.startswith("pipeline_fold_") and f.endswith(".pkl")
    ]

    if not model_files:
        raise FileNotFoundError("No trained models found in working directory.")

    # Sort for deterministic processing order
    model_files.sort()

    logger.info(f"Aggregating predictions from {len(model_files)} models...")

    for model_file in model_files:
        path = os.path.join(models_dir, model_file)
        # logger.info(f"Processing model: {model_file}")
        pipeline = joblib.load(path)

        # Predict on expanded test set (N*3 samples)
        # Output shape: (N*3, n_classes)
        probs_expanded = pipeline.predict_proba(X_test)

        # Aggregate Centroids: (N*3, C) -> (N, 3, C) -> (N, C)
        # The data is structured as [ID1_C1, ID1_C2, ID1_C3, ID2_C1, ...]
        # We reshape to group the 3 centroids per image and average them.
        probs_reshaped = probs_expanded.reshape(n_samples, 3, n_classes)
        probs_mean = probs_reshaped.mean(axis=1)

        # Add to ensemble sum
        avg_probs += probs_mean

    # Average over the number of models (folds)
    avg_probs /= len(model_files)

    # 5. Create and Save Submission
    df_sub = pd.DataFrame(avg_probs, columns=classes)
    df_sub.insert(0, "id", ids)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
