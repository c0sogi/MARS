import pandas as pd
from library.config import Config
from library.utils import setup_logger, timer
from library.data_loader import NQDataset
from library.modeling import LongAnswerClassifier

# Initialize logger
logger = setup_logger("training")


class Trainer:
    """
    Orchestrates the model training workflow for the Natural Questions task.
    """

    def __init__(self, config: Config):
        """
        Initialize the Trainer.

        Args:
            config (Config): Global configuration object.
        """
        self.config = config

    def run(
        self,
        debug_sample_size: int = None,
        num_boost_round: int = None,
        force_reload_data: bool = False,
    ) -> None:
        """
        Executes the full training pipeline: data loading, model fitting, and evaluation.

        Args:
            debug_sample_size (int, optional): If set, enables DEBUG mode and limits
                                               the number of samples for training/validation.
            num_boost_round (int, optional): Override the number of boosting rounds in config.
            force_reload_data (bool): If True, bypasses the cache and re-processes raw data.
        """
        # 1. Apply Hyperparameter Overrides
        if debug_sample_size is not None:
            logger.info(
                f"Overriding sample sizes to {debug_sample_size} and enabling DEBUG mode."
            )
            self.config.DEBUG = True
            self.config.TRAIN_SAMPLE_SIZE = debug_sample_size
            self.config.VAL_SAMPLE_SIZE = debug_sample_size

        if num_boost_round is not None:
            logger.info(f"Overriding NUM_BOOST_ROUND to {num_boost_round}.")
            self.config.NUM_BOOST_ROUND = num_boost_round

        # Determine caching behavior
        # If force_reload_data is True, we pass load_cached_data=False to the loader
        # Otherwise, we use the default from config
        load_cached = self.config.LOAD_CACHED_DATA
        if force_reload_data:
            logger.info("Force reload requested. Ignoring existing cache.")
            load_cached = False

        # 2. Load Data
        # NQDataset handles reading raw JSONL, negative subsampling (for train),
        # flattening, and feature engineering.
        logger.info("Initializing data loaders...")

        with timer("Loading Training Data", logger):
            train_loader = NQDataset(self.config, split="train")
            train_df = train_loader.flatten_and_featurize(load_cached_data=load_cached)

        with timer("Loading Validation Data", logger):
            val_loader = NQDataset(self.config, split="val")
            val_df = val_loader.flatten_and_featurize(load_cached_data=load_cached)

        # Sanity Check
        if train_df.empty:
            raise ValueError(
                "Training dataset is empty. Check input files or sampling logic."
            )
        if val_df.empty:
            raise ValueError(
                "Validation dataset is empty. Check input files or sampling logic."
            )

        logger.info(f"Training set shape: {train_df.shape}")
        logger.info(f"Validation set shape: {val_df.shape}")

        # 3. Initialize Model
        classifier = LongAnswerClassifier(self.config)

        # 4. Train and Evaluate
        # The classifier handles the training loop (epochs/rounds), evaluation metrics,
        # early stopping, and saving the model artifact.
        with timer("Model Training and Evaluation", logger):
            classifier.train(train_df, val_df)

        logger.info("Training pipeline completed successfully.")
