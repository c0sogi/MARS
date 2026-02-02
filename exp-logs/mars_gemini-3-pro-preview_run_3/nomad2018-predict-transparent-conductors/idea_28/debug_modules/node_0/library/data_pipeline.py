import os
import pandas as pd
import logging
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    WORKING_DIR,
)
from library.structure_utils import load_xyz
from library.features import extract_features_from_atoms

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_dataset(metadata_df):
    """
    Iterates over the metadata DataFrame and computes features for each structure.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path' columns.

    Returns:
        pd.DataFrame: DataFrame containing extracted features and 'id'.
    """
    data_list = []
    total = len(metadata_df)

    logger.info(f"Processing {total} samples...")

    for i, row in metadata_df.iterrows():
        file_path = row["file_path"]
        sample_id = row["id"]

        try:
            # Load atomic structure
            atoms = load_xyz(file_path)

            # Extract features using the library function (BVS, RDF, Geometric, Macro)
            feats = extract_features_from_atoms(atoms)

            # Ensure ID is preserved for merging
            feats["id"] = sample_id

            data_list.append(feats)

        except Exception as e:
            logger.warning(f"Error processing sample {sample_id} at {file_path}: {e}")
            continue

    # Create DataFrame from list of dictionaries
    df_features = pd.DataFrame(data_list)
    return df_features


def generate_features(data_type="train", load_cached_data=True):
    """
    Main function to generate or load features.

    Args:
        data_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (X, y)
            - X (pd.DataFrame): Feature matrix.
            - y (pd.DataFrame or None): Target variables if available.
    """
    # Determine file paths based on data type
    if data_type == "train":
        meta_path = TRAIN_METADATA_PATH
        feat_path = TRAIN_FEATURES_PATH
    elif data_type == "val":
        meta_path = VAL_METADATA_PATH
        feat_path = VAL_FEATURES_PATH
    elif data_type == "test":
        meta_path = TEST_METADATA_PATH
        feat_path = TEST_FEATURES_PATH
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    df_features = None

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(feat_path):
        logger.info(f"Loading cached features from {feat_path}")
        try:
            df_features = pd.read_parquet(feat_path)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing features...")
            df_features = None

    # 2. Compute from Scratch if needed
    if df_features is None:
        logger.info(f"Computing features for {data_type} set from scratch...")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)
        df_features = process_dataset(df_meta)

        # Save to cache
        logger.info(f"Saving features to {feat_path}")
        df_features.to_parquet(feat_path, index=False)

    # 3. Prepare X and y
    if data_type in ["train", "val"]:
        # Load metadata to get targets
        df_meta = pd.read_csv(meta_path)

        # Merge features with targets based on ID
        # This ensures alignment even if some samples were skipped during processing
        df_merged = df_features.merge(
            df_meta[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]],
            on="id",
            how="inner",  # Use inner join to keep only samples with both features and targets
        )

        y = df_merged[["formation_energy_ev_natom", "bandgap_energy_ev"]]
        X = df_merged.drop(
            columns=["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        )

        return X, y
    else:
        # For test set, just return features (drop ID for model input)
        X = df_features.drop(columns=["id"])
        # Keep track of IDs for submission if needed, but return X as model input
        return X, None
