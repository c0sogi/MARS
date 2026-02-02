import os
import json
import random
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, load_or_create_cache
from library.feature_engineering import FeatureGenerator

# Initialize logger
logger = setup_logger("data_loader")


class NQDataset:
    """
    Handles loading, preprocessing, and flattening of the Natural Questions dataset.
    """

    def __init__(self, config: Config, split: str = "train"):
        """
        Initialize the dataset loader.

        Args:
            config (Config): Global configuration object.
            split (str): Dataset split ('train', 'val', 'test').
        """
        self.config = config
        self.split = split

        if split == "train":
            self.metadata_path = config.TRAIN_META_PATH
            self.data_path = config.TRAIN_DATA_PATH
        elif split == "val":
            self.metadata_path = config.VAL_META_PATH
            self.data_path = config.TRAIN_DATA_PATH  # Val is subset of train file
        elif split == "test":
            self.metadata_path = config.TEST_META_PATH
            self.data_path = config.TEST_DATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        self.feature_generator = FeatureGenerator(config)

    def _read_json_at_offset(self, file_handle, offset):
        """Reads and parses a single JSON line at a specific byte offset."""
        file_handle.seek(offset)
        line = file_handle.readline()
        if not line:
            return None
        return json.loads(line)

    def _process_raw_data(self) -> pd.DataFrame:
        """
        Reads raw JSONL data using metadata offsets, flattens the structure,
        extracts text, generates labels, and performs negative subsampling for training.

        Returns:
            pd.DataFrame: Flattened dataframe with one row per candidate.
        """
        logger.info(f"Processing raw data for split: {self.split}")

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        meta_df = pd.read_parquet(self.metadata_path)

        # Debugging: Limit sample size
        if self.config.DEBUG:
            limit = (
                self.config.TRAIN_SAMPLE_SIZE
                if self.split == "train"
                else self.config.VAL_SAMPLE_SIZE
            )
            if limit and len(meta_df) > limit:
                logger.info(f"DEBUG mode: Sampling {limit} records from metadata.")
                meta_df = meta_df.sample(n=limit, random_state=self.config.SEED)

        records = []

        with open(self.data_path, "rb") as f:
            for _, row in meta_df.iterrows():
                data = self._read_json_at_offset(f, row["byte_offset"])
                if data is None:
                    continue

                doc_tokens = data["document_text"].split()
                question_text = data["question_text"]
                candidates = data["long_answer_candidates"]

                # Parse annotations for labels (Train/Val only)
                correct_ranges = set()
                if self.split in ["train", "val"]:
                    # annotations is a JSON string in metadata, but we have the raw object here from data file
                    # The raw data file has 'annotations' list directly.
                    # However, the metadata file has 'annotations' column which is a string.
                    # We can use either. Using raw data object is safer for consistency.
                    raw_anns = data.get("annotations", [])
                    for ann in raw_anns:
                        la = ann.get("long_answer", {})
                        st = la.get("start_token", -1)
                        et = la.get("end_token", -1)
                        if st != -1 and et != -1:
                            correct_ranges.add((st, et))

                # Flatten candidates
                temp_candidates = []
                for idx, cand in enumerate(candidates):
                    start = cand["start_token"]
                    end = cand["end_token"]
                    is_top_level = cand["top_level"]

                    # Extract candidate text
                    # Safety check for indices
                    c_text = " ".join(doc_tokens[start:end])

                    # Determine Label
                    label = 0
                    if (start, end) in correct_ranges:
                        label = 1

                    cand_dict = {
                        "example_id": row["example_id"],
                        "candidate_index": idx,
                        "question_text": question_text,
                        "candidate_text": c_text,
                        "start_token": start,
                        "end_token": end,
                        "top_level": is_top_level,
                        "label": label,
                        "document_url": row.get(
                            "document_url", ""
                        ),  # Pass through for grouping
                    }
                    temp_candidates.append(cand_dict)

                # Subsampling for Training
                if self.split == "train":
                    positives = [c for c in temp_candidates if c["label"] == 1]
                    negatives = [c for c in temp_candidates if c["label"] == 0]

                    # If no positive answer, we still might want to train on negatives (No Answer task)
                    # But standard ranking usually requires at least one positive or we treat all as neg.
                    # Strategy: Keep all positives. Sample negatives.

                    n_pos = len(positives)
                    n_neg_keep = max(1, int(n_pos * self.config.NEGATIVE_RATIO))

                    # If no positives, we just sample a few negatives to teach the model "No Answer"
                    if n_pos == 0:
                        n_neg_keep = self.config.NEGATIVE_RATIO

                    if len(negatives) > n_neg_keep:
                        negatives = random.sample(negatives, n_neg_keep)

                    records.extend(positives + negatives)
                else:
                    # Keep all candidates for Val/Test
                    records.extend(temp_candidates)

        return pd.DataFrame(records)

    def flatten_and_featurize(self, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Loads data, flattens it, and generates features.
        Implements strict caching for the flattened dataframe.

        Args:
            load_cached_data (bool): Whether to load from cache.

        Returns:
            pd.DataFrame: Dataframe with features and labels ready for training/inference.
        """
        # 1. Load or Create Flattened Data (Text + Metadata)
        flat_cache_name = f"{self.split}_flattened.parquet"
        flat_cache_path = self.config.get_cache_path(flat_cache_name)

        def _process_flat():
            return self._process_raw_data()

        df_flat = load_or_create_cache(
            file_path=flat_cache_path,
            process_fn=_process_flat,
            load_cached_data=load_cached_data,
            file_type="parquet",
        )

        if df_flat.empty:
            logger.warning(f"Flattened dataframe for {self.split} is empty.")
            return df_flat

        # 2. Generate Features (FeatureGenerator handles its own caching)
        # We pass the flattened dataframe to the feature generator.
        # The generator caches based on split name.
        df_features = self.feature_generator.process_and_cache_features(
            df_flat, split_name=self.split, load_cached_data=load_cached_data
        )

        # 3. Concatenate Features with Metadata/Labels
        # Ensure indices align (they should if caching logic is consistent)
        # We drop text columns from the final training set to save memory if needed,
        # but keeping them is useful for inference extraction.

        # Combine: df_flat contains text/meta/labels. df_features contains numerical features.
        # We assume index alignment is preserved.

        result_df = pd.concat([df_flat, df_features], axis=1)

        # Handle duplicate columns if any (e.g. start_token might be in both)
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        logger.info(f"Final dataset shape for {self.split}: {result_df.shape}")
        return result_df

    def get_data(self):
        """
        Public interface to get the processed data.
        Uses the configuration to determine if cache should be loaded.
        """
        return self.flatten_and_featurize(load_cached_data=self.config.LOAD_CACHED_DATA)
