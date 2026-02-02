import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The RMSE value.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def create_stratified_folds(
    df,
    n_folds: int = Config.N_FOLDS,
    n_bins: int = Config.STRATIFY_BINS,
    seed: int = Config.SEED,
):
    """
    Creates stratified K-Fold indices based on binned 'Pawpularity' target.

    Args:
        df (pd.DataFrame): DataFrame containing the 'Pawpularity' column.
        n_folds (int): Number of folds. Defaults to Config.N_FOLDS.
        n_bins (int): Number of bins for stratification. Defaults to Config.STRATIFY_BINS.
        seed (int): Random seed for shuffling. Defaults to Config.SEED.

    Returns:
        list: A list of tuples (train_index, val_index) where each index is a numpy array.
    """
    # Create a temporary copy to avoid modifying the original dataframe
    df_temp = df.copy()

    # Create bins for continuous target stratification
    # We use pd.cut to discretize the continuous variable 'Pawpularity'
    df_temp["stratify_bin"] = pd.cut(df_temp["Pawpularity"], bins=n_bins, labels=False)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Generate folds based on the bins
    # skf.split returns a generator, we convert it to a list of tuples
    folds = list(skf.split(df_temp, df_temp["stratify_bin"]))

    return folds
