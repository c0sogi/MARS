import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger
from library.data_loader import load_data
from library.embedding_manager import EmbeddingManager


class FeatureEngineer:
    """
    Orchestrates the data loading, feature extraction, and embedding generation
    processes to produce a unified feature set for the HAMF-ADBE model.
    """

    def __init__(self):
        self.logger = setup_logger("feature_engineering")
        self.embedding_manager = EmbeddingManager()

    def build_feature_set(self, load_cached_data: bool = True):
        """
        Constructs the complete feature set for training, validation, and testing.

        Args:
            load_cached_data (bool): Whether to use cached data/embeddings.

        Returns:
            dict: A dictionary containing 'train', 'val', and 'test' keys.
                  Each key maps to a dictionary with:
                  - 'y': Target labels (train/val only)
                  - 'anchor_title': Embeddings (MiniLM)
                  - 'anchor_body': Embeddings (MiniLM)
                  - 'aux_global': Embeddings (MPNet Global)
                  - 'aux_hook': Embeddings (MPNet Title)
                  - 'metadata': Numeric metadata array
                  - 'request_id': Request IDs (test only)
        """
        self.logger.info("Starting feature set construction...")

        # 1. Load DataFrames (Train, Val, Test)
        # Note: data_loader handles the caching of the merged DataFrames
        train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

        # 2. Prepare Training Data for Embedding Generation
        # The Config defines a single cache path for 'train' embeddings.
        # To maintain consistency and allow the EmbeddingManager to cache the full set,
        # we concatenate train and val. The order (Train then Val) is deterministic.
        full_train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        n_train = len(train_df)

        self.logger.info(f"Combined Train+Val shape: {full_train_df.shape}")

        # 3. Get Embeddings
        # Train Split (includes Val)
        self.logger.info("Retrieving embeddings for Train/Val set...")
        tr_anc_title, tr_anc_body, tr_aux_global, tr_aux_hook = (
            self.embedding_manager.get_embeddings(
                full_train_df, split="train", load_cached_data=load_cached_data
            )
        )

        # Test Split
        self.logger.info("Retrieving embeddings for Test set...")
        te_anc_title, te_anc_body, te_aux_global, te_aux_hook = (
            self.embedding_manager.get_embeddings(
                test_df, split="test", load_cached_data=load_cached_data
            )
        )

        # 4. Extract Metadata
        self.logger.info("Extracting numerical metadata...")
        train_meta = self._extract_metadata(train_df)
        val_meta = self._extract_metadata(val_df)
        test_meta = self._extract_metadata(test_df)

        # 5. Assemble Output Dictionary
        # Slice the 'full_train' embeddings back into train and val sets
        feature_set = {
            "train": {
                "y": train_df["requester_received_pizza"].values.astype(int),
                "anchor_title": tr_anc_title[:n_train],
                "anchor_body": tr_anc_body[:n_train],
                "aux_global": tr_aux_global[:n_train],
                "aux_hook": tr_aux_hook[:n_train],
                "metadata": train_meta,
            },
            "val": {
                "y": val_df["requester_received_pizza"].values.astype(int),
                "anchor_title": tr_anc_title[n_train:],
                "anchor_body": tr_anc_body[n_train:],
                "aux_global": tr_aux_global[n_train:],
                "aux_hook": tr_aux_hook[n_train:],
                "metadata": val_meta,
            },
            "test": {
                "request_id": test_df["request_id"].values,
                "anchor_title": te_anc_title,
                "anchor_body": te_anc_body,
                "aux_global": te_aux_global,
                "aux_hook": te_aux_hook,
                "metadata": test_meta,
            },
        }

        self.logger.info("Feature set construction complete.")
        return feature_set

    def _extract_metadata(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts and cleans numerical metadata features.
        """
        # Select columns defined in Config
        meta = df[Config.NUMERIC_COLS].copy()

        # Simple imputation for safety (though data analysis showed no missing values in these cols)
        # We use 0 as a safe default for counts/timestamps in this context
        meta = meta.fillna(0)

        return meta.values.astype(np.float32)
