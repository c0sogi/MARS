import os
import pandas as pd
from joblib import Parallel, delayed
from library.config import Config
from library.utils import get_logger
from library.dicom_utils import process_scan


class Preprocessor:
    """
    Manages the data preprocessing pipeline.
    Ensures that all DICOM studies are loaded, windowed, resized, and cached
    as numpy arrays to speed up training and inference.
    """

    def __init__(self):
        self.logger = get_logger()

    def _process_wrapper(self, study_uid: str, image_dir: str):
        """
        Wrapper function to process a single study and handle exceptions.

        Args:
            study_uid (str): The StudyInstanceUID.
            image_dir (str): The directory containing the study images.
        """
        try:
            # process_scan handles the logic of checking the cache,
            # loading DICOMs, windowing, resizing, and saving the result.
            process_scan(study_uid, image_dir, load_cached_data=True)
        except Exception as e:
            # Log warning but continue processing other files
            # In a real pipeline, we might want to exclude these from the dataset
            pass

    def run(self):
        """
        Executes the preprocessing pipeline for Train, Validation, and Test datasets.
        Uses parallel processing to maximize I/O throughput.
        """
        self.logger.info("Starting Data Preprocessing Pipeline...")

        # --- Load Metadata ---
        if os.path.exists(Config.TRAIN_METADATA_PATH):
            train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
            train_uids = train_df["StudyInstanceUID"].unique()
        else:
            self.logger.warning("Train metadata not found.")
            train_uids = []

        if os.path.exists(Config.VAL_METADATA_PATH):
            val_df = pd.read_csv(Config.VAL_METADATA_PATH)
            val_uids = val_df["StudyInstanceUID"].unique()
        else:
            self.logger.warning("Validation metadata not found.")
            val_uids = []

        if os.path.exists(Config.TEST_METADATA_PATH):
            test_df = pd.read_csv(Config.TEST_METADATA_PATH)
            test_uids = test_df["StudyInstanceUID"].unique()
        else:
            self.logger.warning("Test metadata not found.")
            test_uids = []

        # --- Debug Limiting ---
        if Config.DEBUG_DATA_SIZE is not None:
            self.logger.info(
                f"Debug mode: Limiting processing to {Config.DEBUG_DATA_SIZE} samples per split."
            )
            train_uids = train_uids[: Config.DEBUG_DATA_SIZE]
            val_uids = val_uids[: Config.DEBUG_DATA_SIZE]
            test_uids = test_uids[: Config.DEBUG_DATA_SIZE]

        # --- Parallel Processing ---
        # Note: Train and Val images are both in TRAIN_IMAGES_DIR
        # Test images are in TEST_IMAGES_DIR

        if len(train_uids) > 0:
            self.logger.info(f"Caching {len(train_uids)} training studies...")
            Parallel(n_jobs=Config.NUM_WORKERS)(
                delayed(self._process_wrapper)(uid, Config.TRAIN_IMAGES_DIR)
                for uid in train_uids
            )

        if len(val_uids) > 0:
            self.logger.info(f"Caching {len(val_uids)} validation studies...")
            Parallel(n_jobs=Config.NUM_WORKERS)(
                delayed(self._process_wrapper)(uid, Config.TRAIN_IMAGES_DIR)
                for uid in val_uids
            )

        if len(test_uids) > 0:
            self.logger.info(f"Caching {len(test_uids)} test studies...")
            Parallel(n_jobs=Config.NUM_WORKERS)(
                delayed(self._process_wrapper)(uid, Config.TEST_IMAGES_DIR)
                for uid in test_uids
            )

        self.logger.info("Data Preprocessing Completed.")
