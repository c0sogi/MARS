import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from library.config import METADATA_DIR, WORKING_DIR, RANDOM_SEED, set_seed
from library.feature_engineering import process_structure

# Ensure reproducibility
set_seed(RANDOM_SEED)


class DatasetLoader:
    def __init__(self, metadata_dir=METADATA_DIR, working_dir=WORKING_DIR):
        self.metadata_dir = metadata_dir
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

    def _load_metadata(self, split):
        """Loads metadata CSV for a given split (train, val, test)."""
        path = os.path.join(self.metadata_dir, f"{split}_metadata.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return pd.read_csv(path)

    def _compute_features_parallel(self, df, n_jobs=-1):
        """Computes features in parallel using process_structure."""
        paths = df["file_path"].tolist()
        ids = df["id"].tolist()

        # Parallel execution
        # process_structure is imported from library.feature_engineering
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_structure)(path) for path in paths
        )

        # Combine results into a DataFrame
        features_df = pd.DataFrame(results)
        features_df["id"] = ids
        return features_df

    def _generate_or_load_features(self, split, load_cached_data=True):
        """
        Generates features or loads them from cache.
        """
        cache_path = os.path.join(self.working_dir, f"{split}_features.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features for {split} from {cache_path}...")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Error loading cache for {split}: {e}. Recomputing...")

        print(f"Computing features for {split}...")
        metadata_df = self._load_metadata(split)

        # Compute features
        features_df = self._compute_features_parallel(metadata_df)

        # Merge with metadata
        # We perform a left join on 'id' to keep metadata columns (targets, etc.)
        combined_df = pd.merge(metadata_df, features_df, on="id", how="left")

        # Save to cache
        print(f"Saving {split} features to {cache_path}...")
        combined_df.to_parquet(cache_path, index=False)

        return combined_df

    def load_all_data(self, load_cached_data=True, clean_data=True):
        """
        High-level function to load train, val, and test datasets with features.
        """
        train_df = self._generate_or_load_features("train", load_cached_data)
        val_df = self._generate_or_load_features("val", load_cached_data)
        test_df = self._generate_or_load_features("test", load_cached_data)

        if clean_data:
            train_df, val_df, test_df = self._clean_data(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def _clean_data(self, train_df, val_df, test_df):
        """
        Removes constant columns and handles missing values.
        """
        print("Cleaning data...")

        # Identify numeric columns
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns

        # 1. Drop constant columns based on training set
        # We exclude targets and id from this check to be safe
        exclude_check = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        cols_to_check = [c for c in numeric_cols if c not in exclude_check]

        std = train_df[cols_to_check].std()
        constant_cols = std[std == 0].index.tolist()

        if constant_cols:
            print(f"Dropping {len(constant_cols)} constant columns.")
            train_df = train_df.drop(columns=constant_cols)
            val_df = val_df.drop(columns=constant_cols, errors="ignore")
            test_df = test_df.drop(columns=constant_cols, errors="ignore")

        # 2. Impute missing values
        # Using 0 for missing values (common for RDF bins if no pairs found, etc.)
        train_df = train_df.fillna(0)
        val_df = val_df.fillna(0)
        test_df = test_df.fillna(0)

        return train_df, val_df, test_df


def create_feature_matrix(split, load_cached_data=True):
    """
    Wrapper function to generate features for a specific split.
    Useful if one only needs a specific dataset.
    """
    loader = DatasetLoader()
    return loader._generate_or_load_features(split, load_cached_data)
