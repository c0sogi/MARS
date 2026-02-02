import pandas as pd
from typing import Optional
from library.config import Config
from library.utils import load_metadata as _load_metadata_utils


def load_metadata(
    split: str = "train", sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Loads the metadata for a specific split (train, val, or test) with optional subsampling.

    This function wraps the utility load_metadata function and adds sampling functionality
    for debugging or quick experimentation. It ensures that any sampling performed is
    deterministic by using the fixed random seed from the configuration.

    Args:
        split (str): The dataset split to load. Options: "train", "val", "test".
        sample_size (int, optional): The number of samples to load. If None, loads the full dataset.
                                     If specified and smaller than the dataset size, the dataset
                                     is sampled deterministically.

    Returns:
        pd.DataFrame: The metadata dataframe containing IDs, features, and file paths.
    """
    # Load the full metadata using the provided utility function
    # The utility function handles path resolution based on the split name
    df = _load_metadata_utils(split)

    # Apply sampling if sample_size is specified and is valid
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        # Use the fixed random seed from Config for reproducibility
        df = df.sample(n=sample_size, random_state=Config.RANDOM_SEED).reset_index(
            drop=True
        )
        print(f"Loaded {split} metadata: Subsampled to {len(df)} samples.")
    else:
        print(f"Loaded {split} metadata: Full dataset ({len(df)} samples).")

    return df
