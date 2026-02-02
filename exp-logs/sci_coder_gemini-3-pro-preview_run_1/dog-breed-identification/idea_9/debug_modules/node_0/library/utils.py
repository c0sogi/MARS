import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_oof_data(oof_df: pd.DataFrame, stage: str, config: Config = None):
    """
    Saves Out-Of-Fold predictions to a parquet file in the working directory.

    Args:
        oof_df (pd.DataFrame): DataFrame containing OOF predictions (logits or probabilities).
        stage (str): Identifier for the training stage (e.g., 'stage_1_teacher').
        config (Config, optional): Configuration object. If None, a new instance is created.
    """
    if config is None:
        config = Config()

    # Ensure the working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    filename = f"oof_{stage}.parquet"
    file_path = os.path.join(config.working_dir, filename)

    # Save using parquet as per requirements (no pickle)
    oof_df.to_parquet(file_path, index=False)
    print(f"OOF data for {stage} saved to {file_path}")


def load_oof_data(stage: str, config: Config = None) -> pd.DataFrame:
    """
    Loads Out-Of-Fold predictions from a parquet file.

    Args:
        stage (str): Identifier for the training stage (e.g., 'stage_1_teacher').
        config (Config, optional): Configuration object. If None, a new instance is created.

    Returns:
        pd.DataFrame: DataFrame containing the loaded OOF predictions.

    Raises:
        FileNotFoundError: If the OOF file for the specified stage does not exist.
    """
    if config is None:
        config = Config()

    filename = f"oof_{stage}.parquet"
    file_path = os.path.join(config.working_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"OOF data file not found at {file_path}. Ensure stage '{stage}' has been run."
        )

    oof_df = pd.read_parquet(file_path)
    return oof_df
