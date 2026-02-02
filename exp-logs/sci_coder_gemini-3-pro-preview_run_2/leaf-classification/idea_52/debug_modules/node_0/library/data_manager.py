import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    MARGIN_COLS,
    SHAPE_COLS,
    TEXTURE_COLS,
    GLOBAL_FEATURE_COLS,
    FLOAT_PRECISION,
    CACHE_DIR,
)
from library.image_processing import extract_morphometrics


class LeafData:
    """
    Manages data loading, splitting, and feature organization for the Leaf Classification task.
    """

    def __init__(self):
        self.class_names = []

    def load_datasets(self, load_cached_data=True):
        """
        Loads train, val, and test datasets including tabular and morphometric features.

        Args:
            load_cached_data (bool): If True, attempts to load morphometrics from cache.

        Returns:
            dict: Dictionary containing 'train', 'val', 'test' data dictionaries and 'classes' list.
                  Each split dictionary contains keys: 'global', 'margin', 'shape', 'texture',
                  'morphometrics', 'ids', and 'y' (for train/val).
        """
        # Ensure cache directory exists (redundant with config but safe)
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Load metadata
        df_train = pd.read_csv(TRAIN_DATA_PATH)
        df_val = pd.read_csv(VAL_DATA_PATH)
        df_test = pd.read_csv(TEST_DATA_PATH)

        # Determine classes from training data
        self.class_names = sorted(df_train["species"].unique())
        class_map = {cls: i for i, cls in enumerate(self.class_names)}

        # Process each split
        train_data = self._process_split(df_train, "train", load_cached_data, class_map)
        val_data = self._process_split(df_val, "val", load_cached_data, class_map)
        test_data = self._process_split(
            df_test, "test", load_cached_data, class_map=None
        )

        return {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "classes": self.class_names,
        }

    def _process_split(self, df, split_name, load_cached_data, class_map=None):
        """
        Internal helper to process a single dataset split.
        """
        # 1. Extract Global Tabular Features
        X_global = df[GLOBAL_FEATURE_COLS].values.astype(FLOAT_PRECISION)

        # 2. Extract Component Views
        X_margin, X_shape, X_texture = self.get_component_views(df)

        # 3. Extract Morphometric Features (Image-based)
        # Delegates to library function which handles caching logic
        X_morph = extract_morphometrics(
            df, split_name, load_cached_data=load_cached_data
        )

        # 4. Construct Data Dictionary
        data = {
            "global": X_global,
            "margin": X_margin,
            "shape": X_shape,
            "texture": X_texture,
            "morphometrics": X_morph,
            "ids": df["id"].values,
        }

        # 5. Handle Labels
        if class_map is not None and "species" in df.columns:
            data["y"] = df["species"].map(class_map).values

        return data

    def get_component_views(self, df):
        """
        Slices the dataframe into Margin, Shape, and Texture component arrays.

        Args:
            df (pd.DataFrame): Dataframe containing the feature columns.

        Returns:
            tuple: (X_margin, X_shape, X_texture) as float64 numpy arrays.
        """
        X_margin = df[MARGIN_COLS].values.astype(FLOAT_PRECISION)
        X_shape = df[SHAPE_COLS].values.astype(FLOAT_PRECISION)
        X_texture = df[TEXTURE_COLS].values.astype(FLOAT_PRECISION)
        return X_margin, X_shape, X_texture
