import os
from library import config
from library.data_processor import TaxiDataProcessor
from library.model import FareRegressor


def train_model(load_cached_data=True, train_sample_size=None, max_iter=None):
    """
    Orchestrates the end-to-end training and submission pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
                                 If False, forces re-processing of raw data.
        train_sample_size (int, optional): Number of rows to sample from the training set.
                                           Useful for debugging or quick iterations.
                                           If None, uses the full dataset.
        max_iter (int, optional): Maximum number of boosting iterations (trees).
                                  If provided, overrides the default in config.MODEL_PARAMS.

    Returns:
        FareRegressor: The trained model object.
    """

    # 1. Configuration Overrides
    # We update the configuration dictionary in memory before the model is initialized.
    if max_iter is not None:
        print(f"Overriding max_iter to {max_iter}")
        config.MODEL_PARAMS["max_iter"] = max_iter

    # 2. Data Processing
    # Initialize the processor which handles loading, cleaning, feature engineering, and caching.
    processor = TaxiDataProcessor()

    print(
        f"Processing data (Cached: {load_cached_data}, Sample Size: {train_sample_size})..."
    )
    train_df, val_df, test_df = processor.process_data(
        load_cached_data=load_cached_data, train_sample_size=train_sample_size
    )

    # 3. Model Initialization
    # Initialize the regressor. It will pick up the (potentially modified) params from config.
    regressor = FareRegressor()

    # 4. Training
    # The fit method handles training and prints the validation RMSE.
    print("Starting model training...")
    regressor.fit(train_df, val_df)

    # 5. Inference and Submission
    # Submission generation is now handled conditionally in runfile.py

    print("Pipeline completed successfully.")
    return regressor
