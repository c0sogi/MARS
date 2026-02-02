import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetaFeatureScaler:
    """
    A utility class to compute and normalize meta-features (text lengths)
    for the Siamese DeBERTa model. Wraps sklearn.preprocessing.StandardScaler.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def _get_features(self, df):
        """
        Extracts raw length features from the DataFrame.

        Features:
        - Prompt Length
        - Response A Length
        - Response B Length

        Args:
            df (pd.DataFrame): Input dataframe containing text columns.

        Returns:
            np.ndarray: Array of shape (N, 3) containing raw lengths.
        """
        # Fill NaNs with empty strings to ensure length calculation works robustly
        prompts = df["prompt"].fillna("").astype(str)
        res_a = df["response_a"].fillna("").astype(str)
        res_b = df["response_b"].fillna("").astype(str)

        len_prompt = prompts.str.len().values.reshape(-1, 1)
        len_a = res_a.str.len().values.reshape(-1, 1)
        len_b = res_b.str.len().values.reshape(-1, 1)

        # Concatenate features horizontally
        return np.hstack([len_prompt, len_a, len_b])

    def fit(self, df):
        """
        Computes features from the dataframe and fits the StandardScaler.

        Args:
            df (pd.DataFrame): DataFrame containing 'prompt', 'response_a', 'response_b'.

        Returns:
            self
        """
        features = self._get_features(df)
        self.scaler.fit(features)
        return self

    def transform(self, df):
        """
        Computes features from the dataframe and normalizes them using the fitted scaler.

        Args:
            df (pd.DataFrame): DataFrame containing 'prompt', 'response_a', 'response_b'.

        Returns:
            np.ndarray: Normalized feature matrix of shape (N, 3).
        """
        features = self._get_features(df)
        return self.scaler.transform(features)

    def fit_transform(self, df):
        """
        Fits the scaler and returns transformed features in one step.

        Args:
            df (pd.DataFrame): DataFrame containing 'prompt', 'response_a', 'response_b'.

        Returns:
            np.ndarray: Normalized feature matrix.
        """
        return self.fit(df).transform(df)
