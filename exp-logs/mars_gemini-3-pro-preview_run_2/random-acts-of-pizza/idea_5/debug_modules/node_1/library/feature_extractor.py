import os
import numpy as np
import pandas as pd
from library import config, utils, data_handler, preprocessor, siamese_trainer

# Setup logger
logger = utils.setup_logger("feature_extractor")


class FeatureEngineer:
    """
    Orchestrates the generation of final feature vectors by combining
    fine-tuned text embeddings with scaled tabular metadata.
    """

    def __init__(self):
        self.working_dir = config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

        # Define cache file paths
        self.cache_files = {
            "X_train": os.path.join(self.working_dir, "X_train_combined.npy"),
            "y_train": os.path.join(self.working_dir, "y_train.npy"),
            "X_val": os.path.join(self.working_dir, "X_val_combined.npy"),
            "y_val": os.path.join(self.working_dir, "y_val.npy"),
            "X_test": os.path.join(self.working_dir, "X_test_combined.npy"),
        }

    def generate_features(self, load_cached_data: bool = True):
        """
        Generates or loads the final feature matrices and target vectors.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test)
        """
        # 1. Check Cache
        if load_cached_data:
            if all(os.path.exists(path) for path in self.cache_files.values()):
                logger.info("Loading combined features from cache...")
                try:
                    X_train = np.load(self.cache_files["X_train"])
                    y_train = np.load(self.cache_files["y_train"])
                    X_val = np.load(self.cache_files["X_val"])
                    y_val = np.load(self.cache_files["y_val"])
                    X_test = np.load(self.cache_files["X_test"])
                    return X_train, y_train, X_val, y_val, X_test
                except Exception as e:
                    logger.warning(
                        f"Failed to load cache: {e}. Regenerating features..."
                    )
            else:
                logger.info("Cache miss. Regenerating features...")
        else:
            logger.info("load_cached_data is False. Regenerating features...")

        # 2. Load Data and Tabular Features
        # This handles loading raw data and scaling numerical columns
        X_train_tab, X_val_tab, X_test_tab = preprocessor.get_scaled_features(
            load_cached_data=load_cached_data
        )

        # We need the DataFrames to get the text and targets
        df_train, df_val, df_test = data_handler.load_datasets(
            load_cached_data=load_cached_data
        )

        # 3. Generate Text Embeddings
        # Initialize FineTuner
        tuner = siamese_trainer.FineTuner()

        # Ensure model is trained (or loaded)
        # This step is crucial: it triggers the Siamese fine-tuning if not already done
        tuner.train(load_cached_data=load_cached_data)

        # Extract text lists
        text_prep = preprocessor.TextPreprocessor()
        train_texts = text_prep.get_texts(df_train)
        val_texts = text_prep.get_texts(df_val)
        test_texts = text_prep.get_texts(df_test)

        # Encode texts
        logger.info("Generating embeddings for training set...")
        X_train_emb = tuner.encode(train_texts)

        logger.info("Generating embeddings for validation set...")
        X_val_emb = tuner.encode(val_texts)

        logger.info("Generating embeddings for test set...")
        X_test_emb = tuner.encode(test_texts)

        # 4. Combine Features
        logger.info("Concatenating embeddings and tabular features...")
        X_train = np.hstack([X_train_emb, X_train_tab])
        X_val = np.hstack([X_val_emb, X_val_tab])
        X_test = np.hstack([X_test_emb, X_test_tab])

        # 5. Extract Targets
        y_train = df_train[config.TARGET_COL].values.astype(int)
        y_val = df_val[config.TARGET_COL].values.astype(int)

        # 6. Save to Cache
        logger.info("Saving combined features to cache...")
        np.save(self.cache_files["X_train"], X_train)
        np.save(self.cache_files["y_train"], y_train)
        np.save(self.cache_files["X_val"], X_val)
        np.save(self.cache_files["y_val"], y_val)
        np.save(self.cache_files["X_test"], X_test)

        return X_train, y_train, X_val, y_val, X_test
