import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.features import FeatureEngineer


class DataManager:
    """
    Manages data loading, splitting, and augmentation for the semi-supervised pipeline.
    Delegates feature engineering to the FeatureEngineer class.
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineer()

    def load_and_preprocess(self, load_cached_data=True):
        """
        Loads and preprocesses the dataset using the FeatureEngineer.

        Args:
            load_cached_data (bool): Whether to attempt loading from Parquet cache.

        Returns:
            tuple: (df_train, df_val, df_test)
        """
        return self.feature_engineer.process_data(load_cached_data=load_cached_data)

    def get_folds(
        self, df, n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.SEED
    ):
        """
        Generates Stratified K-Fold indices for cross-validation.

        Args:
            df (pd.DataFrame): Training data containing the target column.
            n_splits (int): Number of folds.
            shuffle (bool): Whether to shuffle before splitting.
            random_state (int): Seed for reproducibility.

        Returns:
            list: List of tuples (train_idx, val_idx).
        """
        target = df[Config.TARGET_COL]
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

        folds = list(skf.split(df, target))
        return folds

    def merge_pseudo_labels(
        self, df_train, df_test, test_probs, threshold=Config.PSEUDO_LABEL_THRESHOLD
    ):
        """
        Identifies high-confidence test predictions and merges them into the training set.

        Args:
            df_train (pd.DataFrame): Original training data.
            df_test (pd.DataFrame): Test data (features only).
            test_probs (np.ndarray): Probability matrix from the Teacher ensemble (N_test, N_classes).
            threshold (float): Confidence threshold to accept a pseudo-label.

        Returns:
            pd.DataFrame: Augmented training dataframe containing original + pseudo-labeled samples.
        """
        # Calculate max probability and predicted class for each test sample
        max_probs = np.max(test_probs, axis=1)
        preds = np.argmax(test_probs, axis=1)

        # Create mask for high-confidence samples
        mask = max_probs >= threshold
        n_pseudo = np.sum(mask)

        print(
            f"Pseudo-Labeling: Found {n_pseudo} samples with confidence >= {threshold}"
        )

        if n_pseudo == 0:
            return df_train

        # Filter test set
        df_pseudo = df_test[mask].copy()

        # Assign predicted labels
        # Note: preds are already in mapped format (0-5), matching df_train
        df_pseudo[Config.TARGET_COL] = preds[mask]

        # Concatenate with original training set
        df_augmented = pd.concat([df_train, df_pseudo], axis=0, ignore_index=True)

        print(f"Augmented Train Set Shape: {df_augmented.shape}")

        return df_augmented
