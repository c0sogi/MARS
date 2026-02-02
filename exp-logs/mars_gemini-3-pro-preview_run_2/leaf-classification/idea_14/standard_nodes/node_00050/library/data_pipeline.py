import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config


class DataPipeline:
    """
    Handles data loading, merging, feature extraction, and preprocessing for the
    Handcrafted Feature Ensemble.
    """

    def __init__(self):
        """
        Initialize the pipeline.
        """
        pass

    def load_and_merge_data(self):
        """
        Loads metadata CSVs and merges the Training and Validation sets
        to maximize sample utilization for the final model.

        Returns:
            tuple: (df_train_full, df_test)
        """
        train_path = os.path.join(Config.METADATA_DIR, "train.csv")
        val_path = os.path.join(Config.METADATA_DIR, "val.csv")
        test_path = os.path.join(Config.METADATA_DIR, "test.csv")

        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            raise FileNotFoundError("One or more metadata files are missing.")

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        # Merge train and validation sets for maximum sample efficiency
        df_train_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)

        return df_train_full, df_test

    def get_handcrafted_features(self, df):
        """
        Extracts the 192 handcrafted features (Margin, Shape, Texture) from the dataframe.

        Args:
            df (pd.DataFrame): The dataframe containing feature columns.

        Returns:
            np.ndarray: Array of shape (N, 192) with float32 precision.
        """
        # Identify feature columns based on keywords
        feature_cols = [
            c for c in df.columns if any(k in c for k in ["margin", "shape", "texture"])
        ]
        # Sort columns to ensure consistent ordering between train and test
        feature_cols.sort()

        return df[feature_cols].values.astype(np.float32)

    def get_image_paths(self, df):
        """
        Constructs absolute file paths for images based on the metadata.

        Args:
            df (pd.DataFrame): Dataframe containing 'image_path' column.

        Returns:
            list: List of full file paths.
        """
        # The metadata 'image_path' is relative to input dir (e.g., 'images/1.jpg')
        # We join it with the INPUT_DIR (e.g., './input')
        return [os.path.join(Config.INPUT_DIR, p) for p in df["image_path"].values]

    def process_view1_handcrafted(self, X_train, X_test):
        """
        Applies StandardScaler to the handcrafted features.

        Args:
            X_train (np.ndarray): Training features.
            X_test (np.ndarray): Test features.

        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def run(self, load_cached_data=True):
        """
        Executes the full data pipeline.

        Args:
            load_cached_data (bool): Unused in this version but kept for signature compatibility.

        Returns:
            dict: Dictionary containing processed data arrays and auxiliary info.
        """
        print("Pipeline: Loading and merging metadata...")
        df_train, df_test = self.load_and_merge_data()

        # --- Target Encoding ---
        print("Pipeline: Encoding targets...")
        le = LabelEncoder()
        y_train = le.fit_transform(df_train["species"])
        classes = le.classes_

        # --- View 1: Handcrafted Features ---
        print("Pipeline: Processing View 1 (Handcrafted)...")
        X_train_hc_raw = self.get_handcrafted_features(df_train)
        X_test_hc_raw = self.get_handcrafted_features(df_test)

        X_train_view1, X_test_view1 = self.process_view1_handcrafted(
            X_train_hc_raw, X_test_hc_raw
        )

        # --- Packaging ---
        data = {
            "X_train_view1": X_train_view1,
            "y_train": y_train,
            "X_test_view1": X_test_view1,
            "test_ids": df_test["id"].values,
            "classes": classes,
        }

        return data
