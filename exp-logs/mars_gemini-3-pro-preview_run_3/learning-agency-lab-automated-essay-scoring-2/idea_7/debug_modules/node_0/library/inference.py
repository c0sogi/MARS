import os
import pandas as pd
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import load_and_preprocess
from library.stacking import inference as stacking_inference


def run_inference(debug=False, load_cached_data=True):
    """
    Main entry point for the inference pipeline.

    Orchestrates the loading of data and the execution of the stacking inference
    workflow defined in the library modules.

    Args:
        debug (bool): Whether to run in debug mode (subsampling data).
        load_cached_data (bool): Whether to load processed data/embeddings from cache.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    # 1. Initialize Configuration
    config = Config()
    config.debug = debug

    # Setup directories
    config.setup()

    # Initialize Logger
    logger = get_logger(os.path.join(config.output_dir, "inference_run.log"))
    logger.info(
        f"Initializing Inference (Debug={debug}, Load Cache={load_cached_data})"
    )

    # 2. Set Random Seed for Reproducibility
    seed_everything(config.seed)

    # 3. Load and Preprocess Data
    # We rely on load_and_preprocess from library.data to handle:
    # - Loading metadata from ./metadata/
    # - Calculating meta-features (word_count, etc.)
    # - Caching the processed dataframe to parquet
    logger.info("Loading test data...")
    # We ignore train_df as we are in inference mode
    _, test_df = load_and_preprocess(config, load_cached_data=load_cached_data)

    logger.info(f"Test data loaded. Shape: {test_df.shape}")

    # 4. Execute Stacking Inference
    # This function (from library.stacking) handles:
    # - Generating/Loading embeddings from the 5 backbone models
    # - Concatenating embeddings with meta-features
    # - Predicting using the 5 LightGBM models
    # - Averaging, clipping, and rounding predictions
    # - Saving the result to ./submission/submission.csv
    logger.info("Running stacking inference pipeline...")
    submission = stacking_inference(test_df, config)

    logger.info("Inference pipeline completed successfully.")
    return submission
