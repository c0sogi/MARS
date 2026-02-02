import os
import numpy as np
import pandas as pd
from library.utils import setup_logger, load_object, WORKING_DIR
from library.data_loader import get_data_splits
from library.feature_extractor import EmbeddingGenerator, extract_metadata_features
from library.pipeline_builder import combine_features


def generate_predictions(
    models_dir=os.path.join(WORKING_DIR, "models"),
    output_dir="./submission",
    n_folds=5,
    load_cached_data=True,
):
    """
    Generates predictions for the test set using the ensemble of trained models.

    Args:
        models_dir (str): Directory containing the saved model files.
        output_dir (str): Directory to save the submission file.
        n_folds (int): Number of folds/models to ensemble.
        load_cached_data (bool): Whether to use cached data/embeddings.
    """
    logger = setup_logger("Inference", os.path.join(WORKING_DIR, "inference.log"))
    logger.info("Starting inference process...")

    # 1. Load Test Data
    # We ignore train/val returns here
    _, _, test_df = get_data_splits(load_cached_data=load_cached_data)

    test_ids = test_df["request_id"].values
    logger.info(f"Loaded test data with {len(test_df)} samples.")

    # 2. Generate/Load Features
    logger.info("Generating features for test set...")
    emb_gen = EmbeddingGenerator()

    # Get Embeddings (High Res + Low Res)
    # process_split handles caching internally based on the split name 'test'
    high_res_test, low_res_test = emb_gen.process_split(
        test_df, "test", load_cached_data=load_cached_data
    )

    # Get Metadata
    meta_test = extract_metadata_features(test_df)

    # Combine
    X_test = combine_features(high_res_test, low_res_test, meta_test)
    logger.info(f"Test feature matrix shape: {X_test.shape}")

    # 3. Ensemble Inference
    logger.info(f"Running inference with {n_folds} models...")

    # Initialize array to store sum of probabilities
    test_preds_sum = np.zeros(len(X_test))

    successful_models = 0

    for fold in range(n_folds):
        model_path = os.path.join(models_dir, f"model_fold_{fold}.joblib")

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}. Skipping fold.")
            continue

        try:
            logger.info(f"Loading model for fold {fold}...")
            model = load_object(model_path)

            # Predict probabilities for the positive class (1)
            # BaggingClassifier.predict_proba returns shape (n_samples, n_classes)
            preds = model.predict_proba(X_test)[:, 1]

            test_preds_sum += preds
            successful_models += 1

        except Exception as e:
            logger.error(f"Error predicting with fold {fold}: {e}")

    if successful_models == 0:
        raise RuntimeError(
            "No models were successfully loaded/executed. Cannot generate submission."
        )

    # 4. Average Predictions
    logger.info(f"Averaging predictions from {successful_models} models...")
    final_preds = test_preds_sum / successful_models

    # 5. Create Submission File
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")

    submission_df = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": final_preds}
    )

    # Ensure strict formatting
    submission_df.to_csv(submission_path, index=False)

    logger.info(f"Submission saved to {submission_path}")
    logger.info("Inference complete.")
