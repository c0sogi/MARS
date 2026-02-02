import os
import numpy as np
import pandas as pd
import json
from library.config import CACHE_DIR, RANDOM_SEED
from library.feature_extractor import extract_features
from library.data_loader import load_metadata


class TargetTransformer:
    """
    Applies log(1+y) transformation to targets and inverts it.
    This helps in stabilizing variance for regression targets that are strictly positive.
    """

    def __init__(self):
        pass

    def transform(self, y):
        """
        Apply log1p transformation: z = log(1 + y)

        Args:
            y (pd.Series or np.array): Original target values.

        Returns:
            np.array: Transformed values.
        """
        return np.log1p(y)

    def inverse_transform(self, z):
        """
        Apply expm1 transformation: y = exp(z) - 1

        Args:
            z (pd.Series or np.array): Transformed target values.

        Returns:
            np.array: Original scale values.
        """
        return np.expm1(z)


class FeatureCleaner:
    """
    Handles missing values and drops constant or zero-variance columns to prevent feature dilution.
    Can save/load its state (columns to drop) to ensure consistency between train and inference.
    """

    def __init__(self):
        self.cols_to_drop = []
        self.fill_value = 0.0
        self.is_fitted = False

    def fit(self, df):
        """
        Identifies constant columns to drop based on the provided DataFrame.

        Args:
            df (pd.DataFrame): Feature matrix (should not include ID or targets).

        Returns:
            self
        """
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Temporarily fill NaNs to check for constant values
        temp_df = numeric_df.fillna(self.fill_value)

        self.cols_to_drop = []
        for col in temp_df.columns:
            # Drop if only one unique value exists (constant column)
            if temp_df[col].nunique() <= 1:
                self.cols_to_drop.append(col)

        self.is_fitted = True
        return self

    def transform(self, df):
        """
        Drops identified constant columns and fills missing values.

        Args:
            df (pd.DataFrame): Feature matrix.

        Returns:
            pd.DataFrame: Cleaned feature matrix.
        """
        # Fill missing values
        # For geometric descriptors, 0.0 is a safe default (e.g. bond count of 0)
        df_clean = df.fillna(self.fill_value)

        # Replace infinity with 0.0 (can occur in density calc if vol is 0)
        df_clean = df_clean.replace([np.inf, -np.inf], 0.0)

        if self.is_fitted:
            # Drop columns that were identified as constant during fit
            # Only drop if they actually exist in the current dataframe
            cols_present = [c for c in self.cols_to_drop if c in df_clean.columns]
            df_clean = df_clean.drop(columns=cols_present)

        return df_clean

    def save(self, filepath):
        """
        Saves the list of columns to drop to a JSON file.
        """
        with open(filepath, "w") as f:
            json.dump(self.cols_to_drop, f)

    def load(self, filepath):
        """
        Loads the list of columns to drop from a JSON file.
        """
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.cols_to_drop = json.load(f)
            self.is_fitted = True
        return self


def get_preprocessed_data(split="train", cleaner=None, load_cached_data=True):
    """
    Orchestrates the data loading, feature extraction, and cleaning pipeline.

    Args:
        split (str): Dataset split ('train', 'val', or 'test').
        cleaner (FeatureCleaner, optional): An instance of FeatureCleaner.
                                          If None and split='train', a new cleaner is fitted.
                                          If None and split!='train', a non-fitted cleaner (impute only) is used.
        load_cached_data (bool): Whether to load cached extracted features from disk.

    Returns:
        tuple: (processed_df, cleaner)
            processed_df (pd.DataFrame): The dataframe with features cleaned and IDs/Targets preserved.
            cleaner (FeatureCleaner): The cleaner instance used (updated if fit).
    """
    # 1. Load Metadata
    meta_df = load_metadata(split=split)

    # 2. Extract Features
    # This step handles caching of the raw extracted features (before cleaning)
    features_df = extract_features(meta_df, split, load_cached_data=load_cached_data)

    # 3. Identify Feature Columns vs Metadata Columns
    # We want to clean only the features, not the ID or Targets
    exclude_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev", "file_path"]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    # 4. Initialize or Use Cleaner
    if cleaner is None:
        cleaner = FeatureCleaner()
        if split == "train":
            # Fit cleaner on training data to identify constant columns
            print(f"Fitting FeatureCleaner on {split} data...")
            cleaner.fit(features_df[feature_cols])
            print(f"Identified {len(cleaner.cols_to_drop)} constant columns to drop.")

    # 5. Apply Cleaning
    # Transform the feature subset
    cleaned_features = cleaner.transform(features_df[feature_cols])

    # 6. Reassemble DataFrame
    # Combine the excluded columns (ID, Targets) with the cleaned features
    rest_df = features_df[[c for c in features_df.columns if c in exclude_cols]]
    processed_df = pd.concat([rest_df, cleaned_features], axis=1)

    return processed_df, cleaner
