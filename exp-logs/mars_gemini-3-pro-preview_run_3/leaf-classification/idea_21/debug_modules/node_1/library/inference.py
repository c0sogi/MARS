import os
import numpy as np
import pandas as pd
import joblib
from library.config import Config
from library.utils import setup_logger
from library.data_processing import OrthogonalDataManager


def _prepare_input(centroid_data, tabular_data):
    """
    Helper to concatenate image centroids and tabular data.

    Args:
        centroid_data (np.ndarray): Image features (N, 2560).
        tabular_data (np.ndarray): Tabular features (N, 192).

    Returns:
        np.ndarray: Concatenated feature matrix (N, 2752).
    """
    return np.hstack([centroid_data, tabular_data])


def predict_test_set(load_cached_data=True):
    """
    Generates predictions for the test set using the trained OS-LDE ensemble.

    Loads trained models for all folds and experts, aggregates predictions via
    averaging, applies post-processing (clipping/normalization), and saves
    the result to submission.csv.

    Args:
        load_cached_data (bool): Whether to use cached feature data. Defaults to True.
    """
    logger = setup_logger("Inference")
    logger.info("Starting inference process...")

    # 1. Load Test Data
    data_manager = OrthogonalDataManager()
    test_data = data_manager.get_data("test", load_cached_data=load_cached_data)

    centroids = test_data["centroids"]
    tabular = test_data["tabular"]
    test_ids = test_data["ids"]

    # 2. Load Label Encoder
    # The label encoder is required to map probability columns to class names
    le_path = os.path.join(Config.WORKING_DIR, "label_encoder.pkl")
    if not os.path.exists(le_path):
        raise FileNotFoundError(
            f"Label encoder not found at {le_path}. Ensure training is complete."
        )

    label_encoder = joblib.load(le_path)
    classes = label_encoder.classes_
    num_classes = len(classes)
    num_samples = len(test_ids)

    logger.info(f"Test data loaded. Samples: {num_samples}, Classes: {num_classes}")

    # 3. Ensemble Prediction Loop
    # Accumulator for final averaged predictions
    final_preds = np.zeros((num_samples, num_classes))

    models_found_count = 0

    # Iterate over all folds
    for fold in range(Config.N_FOLDS):
        fold_preds = np.zeros((num_samples, num_classes))
        fold_models_found = 0

        # Iterate over all experts (A, B, C)
        for expert_name in ["A", "B", "C"]:
            model_filename = f"model_fold_{fold}_expert_{expert_name}.pkl"
            model_path = os.path.join(Config.WORKING_DIR, model_filename)

            if not os.path.exists(model_path):
                logger.warning(f"Model file {model_filename} not found. Skipping.")
                continue

            # Load the expert pipeline
            try:
                pipeline = joblib.load(model_path)
            except Exception as e:
                logger.error(f"Failed to load model {model_filename}: {e}")
                continue

            # Prepare input: Specific Centroid + Shared Tabular
            X_test = _prepare_input(centroids[expert_name], tabular)

            # Predict
            probs = pipeline.predict_proba(X_test)
            fold_preds += probs
            fold_models_found += 1
            models_found_count += 1

        # Average predictions for this fold across the available experts
        if fold_models_found > 0:
            fold_preds /= fold_models_found
            final_preds += fold_preds
        else:
            logger.warning(
                f"No experts found for Fold {fold}. Skipping this fold in ensemble."
            )

    if models_found_count == 0:
        raise RuntimeError("No trained models were found. Cannot generate submission.")

    # Average across all folds
    # Note: We divide by N_FOLDS assuming all folds contributed.
    # If a fold was completely missing, this acts as a regularizer (dampening),
    # but ideally all folds should be present.
    final_preds /= Config.N_FOLDS

    # 4. Post-processing
    # Clip probabilities to avoid log(0) issues in metric calculation
    final_preds = np.clip(final_preds, Config.PROB_CLIP_MIN, Config.PROB_CLIP_MAX)

    # Normalize rows to sum to 1
    row_sums = final_preds.sum(axis=1, keepdims=True)
    final_preds /= row_sums

    # 5. Generate Submission File
    logger.info(f"Generating submission file at {Config.SUBMISSION_PATH}...")

    df_sub = pd.DataFrame(final_preds, columns=classes)
    df_sub.insert(0, "id", test_ids.astype(int))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info("Submission generation complete.")
