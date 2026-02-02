import pandas as pd
import numpy as np
import os
import sys
from library.utils import load_metadata
from library.features import GeometricFeatureExtractor
from library.odometry import run_odometry_processing

# Attempt to silence tqdm to comply with output requirements
try:
    import tqdm as tqdm_module

    # Patch the init to disable it by default
    original_init = tqdm_module.tqdm.__init__

    def new_init(self, *args, **kwargs):
        kwargs["disable"] = True
        original_init(self, *args, **kwargs)

    tqdm_module.tqdm.__init__ = new_init
except ImportError:
    pass


class DriveLoader:
    """
    Orchestrates data loading and preprocessing streams for GNSS positioning.
    Manages metadata, ML features (Stream A), and Odometry constraints (Stream B).
    """

    def __init__(self, cache_dir="./working/idea_17"):
        """
        Args:
            cache_dir (str): Directory to store cached parquet files.
                             Defaults to ./working/idea_17 as required by dependencies.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.feature_extractor = GeometricFeatureExtractor(cache_dir=self.cache_dir)

    def _align_timestamps(self, df):
        """
        Ensures the timestamp column is consistently named 'UnixTimeMillis'.
        GeometricFeatureExtractor returns 'utcTimeMillis', while others use 'UnixTimeMillis'.
        """
        if df is not None and "utcTimeMillis" in df.columns:
            df = df.rename(columns={"utcTimeMillis": "UnixTimeMillis"})
        return df

    def get_data(self, split="train", load_cached_data=True):
        """
        Loads metadata, features, and odometry for a specific split.

        Args:
            split (str): Dataset split ('train', 'val', or 'test').
            load_cached_data (bool): If True, attempts to load processed data from cache.
                                     If False or load fails, recomputes and caches.

        Returns:
            tuple: (metadata_df, features_df, odometry_df)
                - metadata_df: The ground truth or submission index.
                - features_df: ML features (Stream A). For train/val, includes targets.
                - odometry_df: Kinematic constraints (Stream B).
        """
        print(f"[{split.upper()}] Loading Metadata...")
        metadata_df = load_metadata(split)

        # --- Stream A: ML Features ---
        # Extracts geometric features projected onto L1/L5 signal bands.
        # For 'train'/'val', this automatically merges with Ground Truth targets.
        print(f"[{split.upper()}] Processing Stream A (Geometric Features)...")
        features_df = self.feature_extractor.extract_features(
            metadata_df, load_cached_data=load_cached_data
        )
        features_df = self._align_timestamps(features_df)

        # --- Stream B: Robust Odometry ---
        # Computes relative displacement using RANSAC-aided TDCP and Doppler.
        # This function handles its own caching in ./working/idea_17/
        print(f"[{split.upper()}] Processing Stream B (Robust Odometry)...")
        odometry_df = run_odometry_processing(
            metadata_df, load_cached_data=load_cached_data
        )
        odometry_df = self._align_timestamps(odometry_df)

        # Verification of alignment
        if features_df is not None and not features_df.empty:
            feat_trips = features_df["tripId"].nunique()
            meta_trips = metadata_df["tripId"].nunique()
            print(
                f"[{split.upper()}] Loaded Features for {feat_trips}/{meta_trips} trips."
            )

        if odometry_df is not None and not odometry_df.empty:
            odom_trips = odometry_df["tripId"].nunique()
            print(f"[{split.upper()}] Loaded Odometry for {odom_trips} trips.")

        return metadata_df, features_df, odometry_df
