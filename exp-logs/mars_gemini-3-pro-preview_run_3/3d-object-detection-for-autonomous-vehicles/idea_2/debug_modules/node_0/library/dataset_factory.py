import os
import pandas as pd
from library.config import Config
from library.feature_engineering import FeatureEngineer


def build_tabular_dataset(mode, load_cached_data=True, max_samples=None):
    """
    Orchestrates the creation of the tabular dataset for a specific mode.

    Args:
        mode (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load the dataset from parquet cache.
                                 If False or cache miss, regenerates the dataset.
        max_samples (int, optional): If provided, limits the input metadata to this number of samples
                                     for debugging purposes. Creates a temporary metadata file and
                                     separate cache file.

    Returns:
        pd.DataFrame: The structured tabular dataset containing geometric features and targets.
    """
    # Ensure reproducibility
    Config.set_seed()

    # 1. Determine configuration based on mode
    if mode == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        base_save_name = "train_dataset"
        is_test = False
    elif mode == "val":
        metadata_path = Config.VAL_METADATA_PATH
        base_save_name = "val_dataset"
        is_test = False
    elif mode == "test":
        metadata_path = Config.TEST_METADATA_PATH
        base_save_name = "test_dataset"
        is_test = True
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'train', 'val', or 'test'.")

    # 2. Handle Subsetting (Debugging)
    # If max_samples is set, we create a temporary metadata subset and use a distinct cache file
    if max_samples is not None:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load full metadata
        df_meta = pd.read_csv(metadata_path)

        # Sample if necessary
        if len(df_meta) > max_samples:
            df_meta = df_meta.sample(n=max_samples, random_state=Config.RANDOM_SEED)

        # Define temporary metadata path in working directory
        subset_meta_filename = f"{mode}_metadata_subset_{max_samples}.csv"
        metadata_path = os.path.join(Config.WORKING_DIR, subset_meta_filename)

        # Save subset metadata
        df_meta.to_csv(metadata_path, index=False)

        # Update save name to prevent overwriting full dataset cache
        save_name = f"{base_save_name}_subset_{max_samples}.parquet"
    else:
        save_name = f"{base_save_name}.parquet"

    # 3. Instantiate Feature Engineer
    engineer = FeatureEngineer()

    # 4. Create or Load Dataset
    # The FeatureEngineer.create_dataset method handles the caching logic:
    # - Checks if save_name exists in Config.WORKING_DIR if load_cached_data is True.
    # - If not, processes the metadata at metadata_path.
    # - Saves the result to save_name.
    df = engineer.create_dataset(
        metadata_path=metadata_path,
        save_name=save_name,
        load_cached_data=load_cached_data,
        is_test=is_test,
    )

    return df
