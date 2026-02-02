import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config
from library.image_extractor import GeometricFeatureExtractor


class DataPipeline:
    """
    Handles data loading, merging, feature extraction, and preprocessing.
    Combines Handcrafted features with Geometric features.
    """

    def __init__(self):
        """
        Initialize the pipeline and the geometric feature extractor.
        """
        self.geo_extractor = GeometricFeatureExtractor()

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

    def process_features(self, X_train, X_test):
        """
        Applies StandardScaler to the features.

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
        """
        print("Pipeline: Loading and merging metadata...")
        df_train, df_test = self.load_and_merge_data()

        # --- Target Encoding ---
        print("Pipeline: Encoding targets...")
        le = LabelEncoder()
        y_train = le.fit_transform(df_train["species"])
        classes = le.classes_

        # --- Handcrafted Features ---
        print("Pipeline: Extracting Handcrafted Features...")
        X_train_hc = self.get_handcrafted_features(df_train)
        X_test_hc = self.get_handcrafted_features(df_test)

        # --- Geometric Features ---
        print("Pipeline: Extracting Geometric Features...")
        train_paths = self.get_image_paths(df_train)
        test_paths = self.get_image_paths(df_test)

        X_train_geo = self.geo_extractor.extract(train_paths)
        X_test_geo = self.geo_extractor.extract(test_paths)

        # --- Merge ---
        print("Pipeline: Merging Features...")
        X_train_full = np.hstack([X_train_hc, X_train_geo])
        X_test_full = np.hstack([X_test_hc, X_test_geo])

        # --- Scaling ---
        print("Pipeline: Scaling Features...")
        X_train_scaled, X_test_scaled = self.process_features(X_train_full, X_test_full)

        # --- Packaging ---
        data = {
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_test": X_test_scaled,
            "test_ids": df_test["id"].values,
            "classes": classes,
        }

        return data
