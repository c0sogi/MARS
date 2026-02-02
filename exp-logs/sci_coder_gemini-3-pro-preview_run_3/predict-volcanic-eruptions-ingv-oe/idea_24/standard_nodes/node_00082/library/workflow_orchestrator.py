import os
import joblib
import pandas as pd
from library.config import WORKING_DIR, SUBMISSION_PATH, N_FOLDS
from library.model_handler import train_ensemble, predict_ensemble
from library.data_manager import get_test_data


def run_cross_validation(load_cached_data=True, debug_size=None):
    """
    Orchestrates the Stratified K-Fold training process and saves the trained models.

    Args:
        load_cached_data (bool): Whether to load features from cache.
        debug_size (int, optional): Limit data size for debugging.
    """
    # Execute training via the model handler
    # This handles data loading, concatenation, and CV splitting internally
    print("Starting Cross-Validation workflow...")
    models = train_ensemble(load_cached_data=load_cached_data, debug_size=debug_size)

    # Ensure working directory exists for model artifacts
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Save each fold's model
    print(f"Saving {len(models)} models to {WORKING_DIR}...")
    for i, model in enumerate(models):
        model_path = os.path.join(WORKING_DIR, f"model_fold_{i}.joblib")
        joblib.dump(model, model_path)
        print(f"Model for fold {i} saved to {model_path}")

    print("Cross-Validation workflow complete.")


def generate_submission(load_cached_data=True, debug_size=None):
    """
    Generates the submission file by loading trained models and predicting on the test set.
    Triggers training if models are not found.

    Args:
        load_cached_data (bool): Whether to load features from cache.
        debug_size (int, optional): Limit data size for debugging.
    """
    # 1. Load Models
    models = []
    models_exist = True

    print("Checking for trained models...")
    for i in range(N_FOLDS):
        model_path = os.path.join(WORKING_DIR, f"model_fold_{i}.joblib")
        if not os.path.exists(model_path):
            models_exist = False
            print(f"Model file missing: {model_path}")
            break

    if not models_exist:
        print("Trained models not found. Triggering training run...")
        run_cross_validation(load_cached_data=load_cached_data, debug_size=debug_size)

    # Load models after ensuring they exist
    print(f"Loading {N_FOLDS} models from {WORKING_DIR}...")
    models = []
    for i in range(N_FOLDS):
        model_path = os.path.join(WORKING_DIR, f"model_fold_{i}.joblib")
        try:
            model = joblib.load(model_path)
            models.append(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    # 2. Load Test Data
    print("Loading test data...")
    X_test, segment_ids = get_test_data(
        load_cached_data=load_cached_data, debug_size=debug_size
    )

    # 3. Generate Predictions
    print("Generating predictions on test set...")
    predictions = predict_ensemble(models, X_test)

    # 4. Create and Save Submission
    submission_df = pd.DataFrame(
        {"segment_id": segment_ids, "time_to_eruption": predictions}
    )

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
