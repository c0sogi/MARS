import os
import pandas as pd
import numpy as np
import ase.io
from joblib import Parallel, delayed
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    MAX_SAMPLES,
    DEBUG,
)
from library.descriptors import StructureDescriptor


def _process_single_row(row, input_dir, descriptor):
    """
    Helper function to process a single row of metadata.
    Reads the geometry file and extracts features.
    """
    try:
        # Construct full path to geometry file
        # row['file_path'] is relative to input_dir, e.g., "train/1/geometry.xyz"
        full_path = os.path.join(input_dir, row["file_path"])

        # Load crystal structure
        atoms = ase.io.read(full_path, format="aims")

        # Extract features
        features = descriptor.extract(atoms)

        # Add ID for merging/indexing
        features["id"] = row["id"]

        return features
    except Exception as e:
        print(f"Error processing ID {row.get('id', 'unknown')}: {e}")
        return None


class MaterialDataset:
    def __init__(self):
        self.descriptor = StructureDescriptor()

    def load_metadata(self, split):
        """
        Loads the metadata CSV for the specified split.

        Args:
            split (str): 'train', 'val', or 'test'.

        Returns:
            pd.DataFrame: The metadata dataframe.
        """
        if split == "train":
            path = TRAIN_METADATA_PATH
        elif split == "val":
            path = VAL_METADATA_PATH
        elif split == "test":
            path = TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        df = pd.read_csv(path)

        # Apply debugging limit if configured
        if DEBUG and MAX_SAMPLES is not None:
            df = df.head(MAX_SAMPLES)

        return df

    def construct_feature_matrix(self, split, load_cached_data=True):
        """
        Constructs the feature matrix for the given split.
        Implements caching logic using Parquet files.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to try loading from cache.

        Returns:
            pd.DataFrame: Feature matrix indexed by 'id'.
        """
        cache_path = os.path.join(CACHE_DIR, f"{split}_features.parquet")

        # 1. Try to load from cache
        if load_cached_data:
            if os.path.exists(cache_path):
                print(f"Loading cached features for {split} from {cache_path}")
                try:
                    features_df = pd.read_parquet(cache_path)
                    return features_df
                except Exception as e:
                    print(f"Failed to load cache: {e}. Recomputing...")
            else:
                print(f"Cache not found for {split}. Computing features...")
        else:
            print(f"Force recompute enabled for {split}.")

        # 2. Compute from scratch
        metadata_df = self.load_metadata(split)
        print(f"Extracting features for {len(metadata_df)} {split} samples...")

        # Convert DataFrame rows to list of dicts for iteration
        rows = metadata_df.to_dict("records")

        # Parallel processing
        # n_jobs=-1 uses all available cores
        results = Parallel(n_jobs=-1)(
            delayed(_process_single_row)(row, INPUT_DIR, self.descriptor)
            for row in rows
        )

        # Filter out None results (failures)
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            raise RuntimeError(f"No features could be extracted for {split} split.")

        # Create DataFrame
        features_df = pd.DataFrame(valid_results)

        # Set ID as index
        if "id" in features_df.columns:
            features_df.set_index("id", inplace=True)

        # 3. Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        features_df.to_parquet(cache_path)
        print(f"Saved features to {cache_path}")

        return features_df
