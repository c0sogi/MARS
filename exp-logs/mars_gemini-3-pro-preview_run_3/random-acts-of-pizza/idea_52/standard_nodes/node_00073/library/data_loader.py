import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import TRAIN_PATH, VAL_PATH, TEST_PATH, SEED, TARGET_COL, NUM_FOLDS
from library.utils import set_seed


def load_dataset(split: str = "train") -> pd.DataFrame:
    """
    Loads the requested dataset split from Parquet files.

    Args:
        split (str): One of 'train', 'val', 'test', or 'full_train'.
                     'full_train' combines 'train' and 'val' for cross-validation
                     or final model training.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the requested file does not exist.
        ValueError: If an invalid split name is provided.
    """
    if split == "train":
        if not os.path.exists(TRAIN_PATH):
            raise FileNotFoundError(f"Train file not found at {TRAIN_PATH}")
        return pd.read_parquet(TRAIN_PATH)

    elif split == "val":
        if not os.path.exists(VAL_PATH):
            raise FileNotFoundError(f"Validation file not found at {VAL_PATH}")
        return pd.read_parquet(VAL_PATH)

    elif split == "test":
        if not os.path.exists(TEST_PATH):
            raise FileNotFoundError(f"Test file not found at {TEST_PATH}")
        return pd.read_parquet(TEST_PATH)

    elif split == "full_train":
        # Load both train and val and concatenate
        # This allows using 100% of the labeled data for CV or final retraining
        df_train = load_dataset("train")
        df_val = load_dataset("val")

        # Verify columns match to ensure safe concatenation
        if not df_train.columns.equals(df_val.columns):
            raise ValueError(
                "Train and Validation columns do not match, cannot concatenate."
            )

        # Concatenate along the index
        df_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)
        return df_full

    else:
        raise ValueError(
            f"Invalid split name: '{split}'. Must be 'train', 'val', 'test', or 'full_train'."
        )


def get_stratified_folds(
    df: pd.DataFrame,
    n_splits: int = NUM_FOLDS,
    target_col: str = TARGET_COL,
    random_state: int = SEED,
):
    """
    Generates stratified K-Fold indices for the dataset.

    Args:
        df (pd.DataFrame): The dataframe containing the target column.
        n_splits (int): Number of folds. Defaults to config.NUM_FOLDS.
        target_col (str): Name of the target column. Defaults to config.TARGET_COL.
        random_state (int): Random seed for reproducibility. Defaults to config.SEED.

    Returns:
        generator: Yields (train_index, val_index) tuples for each fold.
    """
    # Ensure reproducibility
    set_seed(random_state)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # Extract target for stratification
    y = df[target_col]

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return skf.split(df, y)
