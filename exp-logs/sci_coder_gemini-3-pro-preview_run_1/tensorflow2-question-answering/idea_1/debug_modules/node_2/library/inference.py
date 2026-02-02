import pandas as pd
from library.config import Config
from library.utils import setup_logger, timer
from library.data_loader import NQDataset
from library.modeling import LongAnswerClassifier, generate_submission

# Initialize logger
logger = setup_logger("inference")


class InferenceEngine:
    """
    Manages the end-to-end prediction pipeline for the test set.
    """

    def __init__(self, config: Config):
        """
        Initialize the InferenceEngine.

        Args:
            config (Config): Global configuration object.
        """
        self.config = config

    def run(
        self,
        debug_sample_size: int = None,
        force_reload_data: bool = False,
    ) -> None:
        """
        Executes the inference pipeline: data loading, model loading, and submission generation.

        Args:
            debug_sample_size (int, optional): If set, enables DEBUG mode and limits
                                               the number of samples for processing.
            force_reload_data (bool): If True, bypasses the cache and re-processes raw data.
        """
        logger.info("Starting inference pipeline...")

        # 1. Apply Hyperparameter Overrides
        if debug_sample_size is not None:
            logger.info(
                f"Overriding sample sizes to {debug_sample_size} and enabling DEBUG mode."
            )
            self.config.DEBUG = True
            # NQDataset uses VAL_SAMPLE_SIZE for non-train splits (including test) in debug mode
            self.config.VAL_SAMPLE_SIZE = debug_sample_size

        # Determine caching behavior
        load_cached = self.config.LOAD_CACHED_DATA
        if force_reload_data:
            logger.info("Force reload requested. Ignoring existing cache.")
            load_cached = False

        # 2. Load Test Data
        # NQDataset handles reading raw JSONL, flattening, and feature engineering.
        logger.info("Initializing test data loader...")
        with timer("Loading Test Data", logger):
            test_loader = NQDataset(self.config, split="test")
            test_df = test_loader.flatten_and_featurize(load_cached_data=load_cached)

        if test_df.empty:
            logger.warning(
                "Test dataset is empty. Submission generation may produce empty results."
            )

        logger.info(f"Test set shape: {test_df.shape}")

        # 3. Initialize and Load Model
        logger.info("Loading trained model...")
        classifier = LongAnswerClassifier(self.config)

        if not classifier.load_model():
            raise FileNotFoundError(
                f"Trained model not found at {self.config.get_cache_path('lgbm_model.txt')}. "
                "Please ensure the training pipeline has been run successfully."
            )

        # 4. Generate Submission
        # This function handles prediction, thresholding, short answer extraction,
        # formatting, and saving to CSV.
        with timer("Generating Submission", logger):
            generate_submission(self.config, classifier, test_df)

        logger.info("Inference pipeline completed successfully.")
