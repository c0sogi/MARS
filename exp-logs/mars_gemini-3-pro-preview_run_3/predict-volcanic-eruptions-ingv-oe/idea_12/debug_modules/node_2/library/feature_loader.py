import pandas as pd
from library.utils import load_metadata
from library.features import generate_features


def build_feature_matrix(
    split: str,
    debug: bool = False,
    sample_size: int = 100,
    load_cached_data: bool = True,
) -> pd.DataFrame:
    """
    Orchestrates the creation of the feature matrix for a given data split.

    This function handles:
    1. Loading the metadata for the requested split.
    2. Subsampling the data if in debug mode.
    3. Delegating feature extraction to the parallelized generator.
    4. Merging the extracted features with target variables (for train/val).

    Args:
        split (str): One of 'train', 'val', or 'test'.
        debug (bool): If True, processes only a subset of the data.
        sample_size (int): Number of segments to process if debug is True.
        load_cached_data (bool): Whether to attempt loading features from disk.

    Returns:
        pd.DataFrame: A DataFrame containing features and (if applicable) targets.
    """
    # 1. Load Metadata
    # This provides the map between segment_ids and file paths
    meta_df = load_metadata(split)

    # 2. Handle Debugging/Sampling
    # If debugging, we slice the metadata to process fewer files
    if debug:
        if sample_size is None:
            sample_size = 100
        # Ensure we don't sample more than available
        n = min(len(meta_df), sample_size)
        meta_df = meta_df.iloc[:n].copy()
        output_name = f"{split}_features_debug"
        print(f"DEBUG MODE: Processing first {n} segments for '{split}' split.")
    else:
        output_name = f"{split}_features"

    # 3. Generate Features
    # This calls the heavy-lifting function from library.features
    # It handles caching internally based on the output_name
    feature_df = generate_features(
        metadata_df=meta_df, output_name=output_name, load_cached_data=load_cached_data
    )

    # 4. Merge Targets (for Train/Val)
    # The feature extraction process returns only segment_id and features.
    # We need to re-attach the time_to_eruption target from the metadata.
    if split in ["train", "val"]:
        # Select only the relevant columns for merging
        targets = meta_df[["segment_id", "time_to_eruption"]]

        # Merge on segment_id
        # Using inner join to ensure alignment, though keys should match perfectly
        final_df = pd.merge(feature_df, targets, on="segment_id", how="inner")

        return final_df

    # For test set, just return the features
    return feature_df
