import os
import glob
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger, load_object
from library.feature_extraction import FeaturePreprocessor

# Initialize logger
logger = setup_logger("inference")


def generate_submission(model_paths=None, load_cached_data=True, debug=False):
    """
    Generates predictions for the test set using the ensemble of trained models
    and saves the submission file.

    Args:
        model_paths (list, optional): List of paths to saved model files. If None,
                                      the function searches for models in the default directory.
        load_cached_data (bool): If True, attempts to load pre-computed test features from cache.
        debug (bool): If True, runs inference on a small subset of the test data.

    Returns:
        None
    """
    logger.info(f"Starting submission generation (Debug={debug})...")

    # 1. Identify Models
    if model_paths is None:
        models_dir = os.path.join(Config.WORKING_DIR, "models")
        search_pattern = os.path.join(models_dir, "model_fold_*.joblib")
        model_paths = glob.glob(search_pattern)
        # Sort to ensure deterministic order of processing
        model_paths.sort()

        if not model_paths:
            error_msg = (
                f"No model files found in {models_dir}. Please run training first."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    logger.info(
        f"Found {len(model_paths)} models for inference: {[os.path.basename(p) for p in model_paths]}"
    )

    # 2. Load Test Data
    # FeaturePreprocessor handles caching internally based on the load_cached argument
    preprocessor = FeaturePreprocessor()
    data = preprocessor.get_data(
        split="test", load_cached=load_cached_data, debug=debug
    )

    X_test = data["X"]
    ids_test = data["ids"]

    logger.info(f"Test data loaded. Shape: {X_test.shape}")

    # 3. Generate Predictions (CV-Bagging)
    # We average the probabilities from all fold models to ensure stability
    avg_probs = np.zeros(len(X_test))

    for path in model_paths:
        logger.info(f"Predicting with model: {os.path.basename(path)}")
        try:
            model = load_object(path)

            # Predict probabilities for the positive class (index 1)
            probs = model.predict_proba(X_test)[:, 1]
            avg_probs += probs

        except Exception as e:
            logger.error(f"Failed to run inference with model {path}: {e}")
            raise

    # Compute average
    avg_probs /= len(model_paths)

    # 4. Create Submission DataFrame
    df_sub = pd.DataFrame(
        {"request_id": ids_test, "requester_received_pizza": avg_probs}
    )

    # 5. Save Submission
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_sub.to_csv(save_path, index=False)
    logger.info(f"Submission saved to {save_path}")

    # 6. Validation Check
    logger.info(f"Generated submission shape: {df_sub.shape}")

    if not debug and os.path.exists(Config.SAMPLE_SUBMISSION):
        try:
            sample_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
            if len(df_sub) != len(sample_df):
                logger.warning(
                    f"Row count mismatch! Generated: {len(df_sub)}, Sample: {len(sample_df)}. "
                    "This is expected if running in debug mode or if test set size changed."
                )
            else:
                logger.info("Submission row count matches sample submission.")
        except Exception as e:
            logger.warning(f"Could not verify against sample submission: {e}")
