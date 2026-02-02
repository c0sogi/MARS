import os
import pandas as pd
import numpy as np
import re
from library.image_processor import ImageProcessor


class DataFactory:
    """
    Manages data loading, feature extraction, and caching for the Leaf Classification task.
    Integrates provided tabular features with extracted morphometric features.
    """

    def __init__(self, metadata_dir="./metadata", cache_dir="./working/idea_56/"):
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define column names for the extracted morphometrics
        self.morph_cols = [f"hu_{i}" for i in range(7)] + [
            "aspect_ratio",
            "solidity",
            "extent",
            "eccentricity",
        ]

    def load_datasets(self, load_cached_data=True):
        """
        Loads the training, validation, and test datasets.

        If cached parquet files exist and load_cached_data is True, loads from disk.
        Otherwise, loads metadata, extracts morphometric features from images,
        merges them with provided features, caches the result, and returns dataframes.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (df_train, df_val, df_test)
        """
        train_cache = os.path.join(self.cache_dir, "data_train.parquet")
        val_cache = os.path.join(self.cache_dir, "data_val.parquet")
        test_cache = os.path.join(self.cache_dir, "data_test.parquet")

        # 1. Try loading from cache
        if load_cached_data:
            if (
                os.path.exists(train_cache)
                and os.path.exists(val_cache)
                and os.path.exists(test_cache)
            ):
                print("Loading combined datasets from cache...")
                df_train = pd.read_parquet(train_cache)
                df_val = pd.read_parquet(val_cache)
                df_test = pd.read_parquet(test_cache)
                return df_train, df_val, df_test

        # 2. Process from scratch
        print("Processing datasets from scratch...")

        # Load metadata
        df_train_meta = pd.read_csv(os.path.join(self.metadata_dir, "train.csv"))
        df_val_meta = pd.read_csv(os.path.join(self.metadata_dir, "val.csv"))
        df_test_meta = pd.read_csv(os.path.join(self.metadata_dir, "test.csv"))

        # Initialize ImageProcessor
        processor = ImageProcessor(cache_dir=self.cache_dir)

        # Extract Morphometrics (returns np.ndarray)
        # Note: ImageProcessor handles its own caching of the raw numpy arrays
        train_morph = processor.extract_morphometrics(
            df_train_meta, "train", load_cached_data=load_cached_data
        )
        val_morph = processor.extract_morphometrics(
            df_val_meta, "val", load_cached_data=load_cached_data
        )
        test_morph = processor.extract_morphometrics(
            df_test_meta, "test", load_cached_data=load_cached_data
        )

        # Merge features
        df_train = self._merge_features(df_train_meta, train_morph)
        df_val = self._merge_features(df_val_meta, val_morph)
        df_test = self._merge_features(df_test_meta, test_morph)

        # Cache the combined dataframes
        print(f"Caching combined datasets to {self.cache_dir}...")
        df_train.to_parquet(train_cache, index=False)
        df_val.to_parquet(val_cache, index=False)
        df_test.to_parquet(test_cache, index=False)

        return df_train, df_val, df_test

    def _merge_features(self, meta_df, morph_array):
        """
        Helper to merge morphometric array into the metadata dataframe.
        """
        # Create DataFrame from morphometrics
        df_morph = pd.DataFrame(morph_array, columns=self.morph_cols)

        # Concatenate. We reset index to be safe, though metadata should be 0-indexed.
        meta_df = meta_df.reset_index(drop=True)
        df_morph = df_morph.reset_index(drop=True)

        # Normalize column names (insert underscores if missing)
        # Cite debug_lesson_5: Validate Programmatically Generated Column Names
        def normalize_col(col):
            match = re.match(r"^(margin|shape|texture)(\d+)$", col)
            if match:
                return f"{match.group(1)}_{match.group(2)}"
            return col

        meta_df.columns = [normalize_col(c) for c in meta_df.columns]

        combined_df = pd.concat([meta_df, df_morph], axis=1)

        # Ensure float64 for all feature columns
        feature_cols = self.get_feature_groups(combined_df)["all_features"]
        for col in feature_cols:
            combined_df[col] = combined_df[col].astype(np.float64)

        return combined_df

    def get_feature_groups(self, df):
        """
        Returns a dictionary containing lists of column names for different feature groups.

        Groups:
            - 'margin': margin_1 ... margin_64
            - 'shape': shape_1 ... shape_64
            - 'texture': texture_1 ... texture_64
            - 'morph': hu_0...hu_6, aspect_ratio, solidity, extent, eccentricity
            - 'global': margin + shape + texture
            - 'all_features': global + morph
        """
        all_cols = df.columns.tolist()

        margin_cols = [c for c in all_cols if c.startswith("margin_")]
        shape_cols = [c for c in all_cols if c.startswith("shape_")]
        texture_cols = [c for c in all_cols if c.startswith("texture_")]

        # Ensure sorting for consistency
        margin_cols.sort(key=lambda x: int(x.split("_")[1]))
        shape_cols.sort(key=lambda x: int(x.split("_")[1]))
        texture_cols.sort(key=lambda x: int(x.split("_")[1]))

        morph_cols = [c for c in self.morph_cols if c in all_cols]

        global_cols = margin_cols + shape_cols + texture_cols
        all_features = global_cols + morph_cols

        return {
            "margin": margin_cols,
            "shape": shape_cols,
            "texture": texture_cols,
            "morph": morph_cols,
            "global": global_cols,
            "all_features": all_features,
        }
