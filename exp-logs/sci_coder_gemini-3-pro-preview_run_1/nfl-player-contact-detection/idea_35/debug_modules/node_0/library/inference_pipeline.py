import os
import pandas as pd
import numpy as np
import gc
from library.config import Config
from library.utils import setup_logger, seed_everything
from library.feature_engineering import generate_features
from library.model_factory import get_model
from library.training_pipeline import get_feature_cols

# Initialize Logger
logger = setup_logger("inference_pipeline")


def run_inference(load_cached_data: bool = True, debug_sample: int = None):
    """
    Executes the inference pipeline:
    1. Loads the sample submission template.
    2. Generates features for the test set (survivors of gating).
    3. Loads the trained Expert Ensemble.
    4. Generates predictions and merges them back to the full template.
    5. Saves the submission file.

    Args:
        load_cached_data (bool): Whether to use cached feature files.
        debug_sample (int): Number of rows to process for debugging.
    """
    seed_everything(Config.SEED)

    logger.info("--- Starting Inference Pipeline ---")

    # 1. Load Submission Template
    # We must provide a prediction for every contact_id in this file.
    logger.info(f"Loading sample submission from {Config.SAMPLE_SUBMISSION_PATH}...")
    df_template = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    if debug_sample:
        logger.info(f"Debug mode: subsetting template to {debug_sample} rows.")
        df_template = df_template.iloc[:debug_sample].copy()

    # 2. Generate Test Features
    # Note: generate_features applies gating, so df_features will likely have fewer rows
    # than df_template. Rows dropped by gating are assumed to be 0 (No Contact).
    logger.info("Generating test features...")
    df_features = generate_features(
        "test", load_cached_data=load_cached_data, debug_sample=debug_sample
    )

    # 3. Prepare Feature Matrix
    feature_cols = get_feature_cols(df_features)
    X_test = df_features[feature_cols]
    contact_ids_survivors = df_features["contact_id"]

    logger.info(f"Features generated. Survivors of gating: {len(X_test)}")

    # 4. Load Models and Predict
    model_types = ["lgbm", "xgb", "catboost"]
    avg_probs = np.zeros(len(X_test))
    models_loaded = 0

    logger.info("Loading Expert Ensemble models...")
    for m_name in model_types:
        model_path = os.path.join(Config.MODEL_DIR, f"expert_{m_name}.joblib")

        if os.path.exists(model_path):
            try:
                logger.info(f"Loading {m_name.upper()} from {model_path}...")
                model = get_model(m_name)
                model.load(model_path)

                # Predict probabilities
                probs = model.predict_proba(X_test)
                avg_probs += probs
                models_loaded += 1
            except Exception as e:
                logger.error(f"Failed to load or predict with {m_name}: {e}")
        else:
            logger.warning(f"Model file not found: {model_path}")

    if models_loaded == 0:
        logger.error("No models loaded. Cannot perform inference.")
        # In a strict pipeline, we might raise an error.
        # For robustness, we might output all zeros, but let's raise here to alert.
        raise RuntimeError("Ensemble inference failed: No models available.")

    # Average the probabilities
    avg_probs /= models_loaded

    # 5. Apply Threshold
    thresh_path = os.path.join(Config.MODEL_DIR, "best_threshold.npy")
    threshold = 0.5  # Default fallback

    if os.path.exists(thresh_path):
        try:
            threshold = np.load(thresh_path)[0]
            logger.info(f"Loaded optimal threshold: {threshold}")
        except Exception as e:
            logger.warning(
                f"Failed to load threshold file: {e}. Using default {threshold}."
            )
    else:
        logger.warning(
            f"Threshold file not found at {thresh_path}. Using default {threshold}."
        )

    # Generate binary predictions
    predictions = (avg_probs >= threshold).astype(int)

    # 6. Merge Predictions with Template
    logger.info("Merging predictions with submission template...")

    # Create a DataFrame of the survivors' predictions
    df_preds = pd.DataFrame(
        {"contact_id": contact_ids_survivors, "contact_pred": predictions}
    )

    # Left join the template with predictions.
    # Rows that were gated out (not in df_preds) will get NaN.
    submission = df_template[["contact_id"]].merge(
        df_preds, on="contact_id", how="left"
    )

    # Fill NaNs with 0 (No Contact) and cast to integer
    fill_count = submission["contact_pred"].isna().sum()
    submission["contact"] = submission["contact_pred"].fillna(0).astype(int)

    # Drop the temporary prediction column
    submission = submission.drop(columns=["contact_pred"])

    logger.info(f"Filled {fill_count} gated-out rows with 0.")

    # 7. Save Submission
    # Ensure directory exists if path contains one (Config path is usually just filename)
    if os.path.dirname(Config.SUBMISSION_OUTPUT_PATH):
        os.makedirs(os.path.dirname(Config.SUBMISSION_OUTPUT_PATH), exist_ok=True)

    submission.to_csv(Config.SUBMISSION_OUTPUT_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_OUTPUT_PATH}")
    logger.info(f"Final Submission Shape: {submission.shape}")

    # Clean up
    del df_template, df_features, X_test, df_preds, submission
    gc.collect()

    logger.info("Inference Pipeline Completed Successfully.")
