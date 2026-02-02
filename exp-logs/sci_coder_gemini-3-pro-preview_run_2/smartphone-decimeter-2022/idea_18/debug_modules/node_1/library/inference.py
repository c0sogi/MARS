import os
import pandas as pd
from library.config import Config
from library.utils import get_logger
from library.data_processing import GNSSPreprocessor
from library.dataset import SKFDataset
from library.trainer import Trainer


class InferenceEngine:
    """
    Orchestrates the inference process: data preparation, model prediction, and submission generation.
    """

    def __init__(self):
        """
        Initialize the InferenceEngine with necessary components.
        """
        self.logger = get_logger("inference_engine")
        self.preprocessor = GNSSPreprocessor()
        # Trainer handles model initialization, checkpoint loading, and prediction logic
        self.trainer = Trainer()

    def generate_submission(self, load_cached_data: bool = True):
        """
        Runs the full inference pipeline to generate the submission file.

        Args:
            load_cached_data (bool): Whether to try loading preprocessed test data from cache.
                                     If False or cache is missing, data will be reprocessed from raw files.
        """
        self.logger.info("Step 1: Preparing Test Data")

        # Process test data using the preprocessor.
        # This handles raw data loading, windowing, scaling (using saved scalers), and caching.
        X_seq, X_sky, _, test_meta = self.preprocessor.process_test(
            load_cached_data=load_cached_data
        )

        if X_seq is None or len(X_seq) == 0:
            self.logger.error("No test data generated. Aborting inference.")
            return

        self.logger.info(f"Test data shape: Sequence={X_seq.shape}, Sky={X_sky.shape}")

        # Wrap the numpy arrays in the PyTorch Dataset
        test_dataset = SKFDataset(X_seq, X_sky, y=None)

        self.logger.info("Step 2: Running Prediction and Generating Submission")

        # The Trainer's predict method encapsulates the inference loop:
        # 1. Loads the best model weights from Config.MODEL_PATH
        # 2. Runs the model in evaluation mode
        # 3. Reconstructs Lat/Lon from predicted metric residuals using WGS84 utils
        # 4. Saves the result to Config.SUBMISSION_PATH
        self.trainer.predict(test_dataset, test_meta)

        self.logger.info(
            f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}"
        )
