import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger, load_model
from library.embedding_manager import generate_embeddings

logger = setup_logger("inference")


def run_inference():
    """
    Executes the inference pipeline:
    1. Loads the test set embeddings (using cached data if available).
    2. Loads the 5 pre-trained fold models.
    3. Generates predictions for the test set using CV-Bagging (averaging).
    4. Saves the submission file.
    """
    logger.info("Starting inference workflow...")

    # 1. Load Data
    # We use generate_embeddings to retrieve X_test.
    # It handles loading from cache or computing from scratch if necessary.
    # We ignore the training/val returns here.
    logger.info("Retrieving test set features...")
    _, _, _, _, X_test, _ = generate_embeddings(load_cached_data=True)

    logger.info(f"Test Data Shape: {X_test.shape}")

    # 2. Initialize Prediction Accumulator
    n_folds = 5  # Matches the trainer.py configuration
    test_preds_sum = np.zeros(len(X_test), dtype=np.float64)
    models_found = 0

    # 3. Iterate through Folds and Predict
    for fold in range(n_folds):
        model_filename = f"model_fold_{fold}.joblib"
        model_path = os.path.join(Config.WORKING_DIR, model_filename)

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}. Skipping fold.")
            continue

        try:
            logger.info(f"Loading model for Fold {fold + 1} from {model_path}...")
            model = load_model(model_path)

            # Predict probabilities (class 1)
            fold_probs = model.predict_proba(X_test)[:, 1]
            test_preds_sum += fold_probs
            models_found += 1

        except Exception as e:
            logger.error(f"Error processing fold {fold}: {e}")

    if models_found == 0:
        raise RuntimeError(
            "No models were successfully loaded. Cannot generate predictions."
        )

    # 4. Average Predictions
    logger.info(f"Averaging predictions from {models_found} models...")
    avg_test_preds = test_preds_sum / models_found

    # 5. Generate Submission File
    logger.info("Loading test metadata for submission alignment...")
    if not os.path.exists(Config.METADATA_TEST):
        raise FileNotFoundError(f"Test metadata not found at {Config.METADATA_TEST}")

    df_test_meta = pd.read_csv(Config.METADATA_TEST)

    # Ensure alignment
    if len(df_test_meta) != len(avg_test_preds):
        raise ValueError(
            f"Mismatch between metadata rows ({len(df_test_meta)}) "
            f"and predictions ({len(avg_test_preds)})"
        )

    # Create submission DataFrame
    submission = pd.DataFrame(
        {
            "request_id": df_test_meta["request_id"],
            "requester_received_pizza": avg_test_preds,
        }
    )

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info("Inference workflow completed successfully.")
