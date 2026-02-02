import os
import lightgbm as lgb
import library.config as config
import library.data_processor as data_processor
import library.trainer as trainer


def load_models(model_dir):
    """
    Loads the ensemble of LightGBM models from the specified directory.

    Args:
        model_dir (str): Directory containing the saved model files.

    Returns:
        list: A list of loaded lightgbm.Booster objects.
    """
    models = []
    # Iterate through the expected number of folds defined in config
    for fold in range(config.NUM_FOLDS):
        model_path = os.path.join(model_dir, f"lgbm_model_fold_{fold}.txt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file for fold {fold} not found at {model_path}"
            )

        # Load the model
        # Note: Loaded Boosters usually have best_iteration=0 (meaning use all trees)
        # unless specifically set, but predict handles this.
        model = lgb.Booster(model_file=model_path)
        models.append(model)

    return models


def predict_and_submit(load_cached_data=True, sample_size=None):
    """
    Loads pre-computed test features and the ensemble of trained models,
    computes the arithmetic mean of predictions across all folds,
    and generates the final submission.csv.

    Args:
        load_cached_data (bool): Whether to attempt loading features from cache.
        sample_size (int, optional): Number of test samples to process (for debugging).
    """
    # 1. Load Test Features
    # Uses the data_processor module which handles caching and parallel extraction
    test_meta_path = os.path.join(config.METADATA_DIR, "test.csv")

    print("Loading test features...")
    test_df = data_processor.process_set(
        metadata_path=test_meta_path,
        cache_name="test_features.parquet",
        load_cached_data=load_cached_data,
        sample_size=sample_size,
    )

    if test_df.empty:
        print("Warning: Test DataFrame is empty. Aborting prediction.")
        return

    # 2. Load Trained Models
    # Models are stored in the working directory defined in config
    print(f"Loading {config.NUM_FOLDS} trained models from {config.WORKING_DIR}...")
    try:
        models = load_models(config.WORKING_DIR)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Ensure that the training pipeline has been run successfully.")
        return

    # 3. Generate Submission
    # Uses the shared logic in trainer.py to average predictions and save the CSV
    print("Generating predictions...")
    trainer.generate_submission(models, test_df)
