import pandas as pd
from typing import Optional, Tuple, Dict
from library.feature_engineering import FeatureEngineer
from library.utils import get_logger

# Initialize logger
logger = get_logger("knowledge_base")


class KnowledgeBase:
    """
    Deterministic memory component for the hybrid normalization system.
    Maps (raw_token, predicted_class) pairs to their normalized text using a
    lookup table constructed from the training data.
    """

    def __init__(self):
        """
        Initializes the KnowledgeBase with an empty lookup table.
        """
        # Dictionary mapping (raw_token, class) -> normalized_text
        self.lookup_table: Dict[Tuple[str, str], str] = {}

    def build(self, load_cached_data: bool = True) -> None:
        """
        Constructs the lookup table from the training data.

        Uses FeatureEngineer to aggregate and cache the (before, class) -> after mapping
        from the training set, then loads it into an optimized in-memory dictionary.

        Args:
            load_cached_data (bool): If True, attempts to load the intermediate DataFrame
                                     from the parquet cache managed by FeatureEngineer.
        """
        logger.info("Initializing Knowledge Base build process...")

        # Delegate data aggregation and caching to FeatureEngineer
        # FeatureEngineer.build_or_load_knowledge_base handles the logic of:
        # 1. Checking if parquet exists (if load_cached_data=True)
        # 2. If not, reading train.csv, grouping by (before, class), finding most frequent 'after'
        # 3. Saving to parquet
        fe = FeatureEngineer()
        kb_df = fe.build_or_load_knowledge_base(load_cached_data=load_cached_data)

        logger.info(
            f"Loaded Knowledge Base DataFrame with {len(kb_df)} rows. Converting to dictionary..."
        )

        # Convert DataFrame to dictionary for O(1) inference lookups
        # Structure: {(before, class): after}
        # Using zip on columns is significantly faster than iterating rows
        # We ensure 'before' and 'class' are treated as strings to match inference types
        keys = zip(kb_df["before"].astype(str), kb_df["class"].astype(str))
        values = kb_df["after"].astype(str)

        self.lookup_table = dict(zip(keys, values))

        logger.info(f"Knowledge Base ready. Total entries: {len(self.lookup_table)}")

    def query(self, token: str, token_class: str) -> Optional[str]:
        """
        Retrieves the normalized text for a given token and class.

        Args:
            token (str): The raw input token (before).
            token_class (str): The predicted class of the token.

        Returns:
            Optional[str]: The normalized text (after) if found, else None.
        """
        return self.lookup_table.get((token, token_class))
