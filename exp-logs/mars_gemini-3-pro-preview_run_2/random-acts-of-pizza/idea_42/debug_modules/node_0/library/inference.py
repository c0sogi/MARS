import os
import numpy as np
import pandas as pd
import joblib
from library import config
from library import utils
from library import data_loader
from library import embedding_manager
from library import trainer

# Import custom transformers to ensure they are available in the namespace during joblib unpickling
from library.custom_transformers import ArraySelector, WhitenedPCANormalizer

# Initialize Logger
logger = utils.setup_logger(
    "inference", os.path.join(config.WORKING_DIR, "inference.log")
)


def generate_submission(load_cached_data=True, debug_mode=config.DEBUG_MODE):
    """
    Generates predictions for the test set using the ensemble of trained models (CV-Bagging).

    The function performs the following steps:
    1. Loads the test dataset and embeddings.
    2. Constructs the fused feature matrix for the test set.
    3. Loads each of the 5 trained fold-models.
    4. Generates probability predictions from each model.
    5. Averages the predictions to reduce variance.
    6. Saves the final predictions to the submission file.

    Args:
        load_cached_data (bool): Whether to load processed data/embeddings from cache.
        debug_mode (bool): Whether to run in debug mode (subset of data).
    """
    utils.set_seed(config.SEED)
    logger.info("Starting inference process...")

    # =========================================================================
    # 1. Load Data and Embeddings
    # =========================================================================
    logger.info("Loading datasets...")
    # We need to load all datasets because embedding_manager expects them to verify/generate embeddings
    train_df, val_df, test_df = data_loader.load_dataset(
        load_cached_data=load_cached_data, debug_mode=debug_mode
    )

    logger.info("Loading embeddings...")
    embeddings = embedding_manager.get_embeddings(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # =========================================================================
    # 2. Build Test Feature Matrix
    # =========================================================================
    logger.info("Building test feature matrix...")
    # Uses the shared utility from trainer to ensure consistent feature construction
    X_test = trainer.build_feature_matrix(test_df, embeddings, "test")

    logger.info(f"Test Feature Matrix Shape: {X_test.shape}")

    # =========================================================================
    # 3. Ensemble Prediction (CV-Bagging)
    # =========================================================================
    logger.info(f"Loading {config.N_FOLDS} fold models for inference...")

    models_dir = os.path.join(config.WORKING_DIR, "models")
    test_preds_accum = np.zeros(X_test.shape[0])

    successful_models = 0

    for fold in range(config.N_FOLDS):
        model_path = os.path.join(models_dir, f"model_fold_{fold}.joblib")

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}. Skipping this fold.")
            continue

        try:
            # Load the pipeline (includes preprocessing and classifier)
            model = joblib.load(model_path)

            # Predict probability of class 1 (Success)
            # The pipeline handles scaling, whitening, and normalization internally
            preds = model.predict_proba(X_test)[:, 1]

            test_preds_accum += preds
            successful_models += 1
            logger.info(f"Fold {fold} predictions generated.")

        except Exception as e:
            logger.error(f"Error loading or predicting with fold {fold} model: {e}")

    if successful_models == 0:
        raise RuntimeError(
            "No models were successfully loaded. Cannot generate submission."
        )

    # Average the predictions
    avg_test_preds = test_preds_accum / successful_models
    logger.info(f"Averaged predictions from {successful_models} models.")

    # =========================================================================
    # 4. Save Submission
    # =========================================================================
    submission_df = pd.DataFrame(
        {
            "request_id": test_df["request_id"],
            "requester_received_pizza": avg_test_preds,
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    logger.info(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)

    # Verification
    logger.info(f"Submission shape: {submission_df.shape}")
    logger.info(f"Submission head:\n{submission_df.head()}")
    logger.info("Inference process completed successfully.")
