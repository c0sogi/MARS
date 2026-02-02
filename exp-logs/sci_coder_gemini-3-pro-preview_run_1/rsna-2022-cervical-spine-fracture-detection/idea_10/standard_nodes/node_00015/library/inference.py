import os
import pandas as pd
from library.config import Config
from library.utils import get_logger, seed_everything
from library.trainers import FractureDetectionTrainer


class InferencePipeline:
    """
    Manages the end-to-end inference pipeline for the test set.
    Orchestrates the loading of models, data processing, and submission generation
    via the FractureDetectionTrainer.
    """

    def __init__(self):
        """
        Initializes the inference pipeline and the underlying trainer/predictor.
        """
        self.logger = get_logger("inference_pipeline")
        self.trainer = FractureDetectionTrainer()

    def run(self, test_metadata_path: str = None, load_cached_data: bool = True):
        """
        Executes the inference pipeline on the test set.

        Args:
            test_metadata_path (str, optional): Path to the test metadata CSV.
                                                Defaults to Config.TEST_METADATA_PATH.
            load_cached_data (bool): If True, attempts to use cached intermediate features.
                                     If False, forces re-extraction of features from images.
        """
        # 1. Setup and Reproducibility
        seed_everything(Config.SEED)
        self.logger.info("Starting Inference Pipeline.")

        # 2. Load Metadata
        if test_metadata_path is None:
            test_metadata_path = Config.TEST_METADATA_PATH

        self.logger.info(f"Loading test metadata from: {test_metadata_path}")
        if not os.path.exists(test_metadata_path):
            raise FileNotFoundError(
                f"Test metadata file not found at {test_metadata_path}"
            )

        test_df = pd.read_csv(test_metadata_path)

        # 3. Handle Debug Mode
        if Config.DEBUG:
            self.logger.info(
                f"DEBUG MODE ENABLED: Subsetting test data to {Config.DEBUG_SAMPLE_SIZE} samples."
            )
            test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        self.logger.info(f"Total studies to process: {len(test_df)}")

        # 4. Manage Caching Logic
        # The trainer's predict_test_set method defaults to using cache if available.
        # To respect load_cached_data=False, we explicitly remove the cache file if it exists.
        if not load_cached_data:
            cache_file = Config.CACHE_FEATURES_TEST
            if os.path.exists(cache_file):
                self.logger.info(
                    f"load_cached_data=False: Removing existing cache file {cache_file} to force re-computation."
                )
                try:
                    os.remove(cache_file)
                except OSError as e:
                    self.logger.warning(f"Failed to remove cache file: {e}")
            else:
                self.logger.info(
                    "load_cached_data=False: No existing cache file found."
                )

        # 5. Execute Prediction
        # This calls the 3-stage pipeline:
        # Stage 1: Segmentation & Global Context (AnatomicalSegmentor)
        # Stage 2: Local Feature Extraction (FractureEncoder)
        # Stage 3: Sequence Aggregation & Prediction (HCHRNAggregator)
        # Finally generates submission.csv
        try:
            self.trainer.predict_test_set(test_df)
            self.logger.info("Inference completed successfully.")
            self.logger.info(f"Submission generated at: {Config.SUBMISSION_FILE}")
        except Exception as e:
            self.logger.error(f"An error occurred during inference: {e}")
            raise e
