import os
import logging
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logging, seed_everything
from library.data_pipeline import DataPipeline
from library.models import TriEnsemble


def generate_submission(threshold=None, load_cached_data=True, debug=False):
    """
    Executes the inference pipeline to generate the final submission file.

    This function loads the trained Expert Ensemble models, processes the test data
    to generate Dual-Basis Time-Domain features, applies the optimized decision
    threshold, and formats the output according to the competition requirements.

    Args:
        threshold (float, optional): The decision threshold to apply. If None, attempts
                                     to load the best threshold from cache.
        load_cached_data (bool): Whether to use cached features and models.
        debug (bool): Whether to run in debug mode (using a data subset).
    """
    # 1. Setup
    setup_logging()
    seed_everything(Config.SEED)
    logger = logging.getLogger(__name__)
    logger.info("Starting Inference Pipeline...")

    # 2. Determine Threshold
    if threshold is None:
        threshold_path = Config.CACHE_BEST_THRESHOLD
        if os.path.exists(threshold_path):
            try:
                threshold = np.load(threshold_path).item()
                logger.info(f"Loaded best threshold from cache: {threshold}")
            except Exception as e:
                logger.error(f"Failed to load threshold from cache: {e}")
                threshold = 0.5
        else:
            logger.warning("No cached threshold found. Defaulting to 0.5.")
            threshold = 0.5
    else:
        logger.info(f"Using provided threshold: {threshold}")

    # 3. Load Expert Models
    logger.info("Loading Expert Ensemble models...")
    ensemble = TriEnsemble(Config)

    # Define paths for expert models
    expert_model_paths = {
        "lgbm": Config.MODEL_EXPERT_LGBM,
        "xgb": Config.MODEL_EXPERT_XGB,
        "cat": Config.MODEL_EXPERT_CAT,
    }

    # Verify existence of at least one model
    available_models = {
        k: v for k, v in expert_model_paths.items() if os.path.exists(v)
    }
    if not available_models:
        logger.error("No trained expert models found at the expected paths.")
        logger.error(
            "Please ensure the training pipeline has been executed successfully."
        )
        return

    ensemble.load_models(expert_model_paths)

    # 4. Load Test Data (Feature Engineering)
    # DataPipeline handles feature generation and caching internally
    data_pipeline = DataPipeline(Config)
    df_test = data_pipeline.load_data(
        mode="test", load_cached_data=load_cached_data, debug=debug
    )

    if df_test.empty:
        logger.warning("Test dataset is empty. Cannot generate predictions.")
        # Fallback: Create dummy submission if sample exists
        if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
            logger.info("Generating empty submission based on sample file.")
            sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
            sub["contact"] = 0
            os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
            sub.to_csv(Config.SUBMISSION_PATH, index=False)
        return

    # 5. Generate Predictions
    logger.info("Generating predictions for test set...")
    probs = ensemble.predict_proba(df_test)
    predictions = (probs > threshold).astype(int)

    # 6. Format Submission
    logger.info("Formatting submission file...")

    # Create prediction dataframe
    pred_df = pd.DataFrame(
        {"contact_id": df_test["contact_id"], "contact": predictions}
    )

    # Merge with sample submission to ensure all IDs are present
    # This handles cases where rows might have been dropped during gating
    sample_path = Config.SAMPLE_SUBMISSION_PATH
    if os.path.exists(sample_path):
        sample_sub = pd.read_csv(sample_path)

        # Left join on sample_sub to preserve all required contact_ids and order
        final_sub = sample_sub[["contact_id"]].merge(
            pred_df, on="contact_id", how="left"
        )

        # Fill missing values (dropped rows are assumed non-contact) with 0
        final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)
    else:
        logger.warning("Sample submission file not found. Using raw predictions.")
        final_sub = pred_df

    # 7. Save Output
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
    logger.info("Inference pipeline completed.")
