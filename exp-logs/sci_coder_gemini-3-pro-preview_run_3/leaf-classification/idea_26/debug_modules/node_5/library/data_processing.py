import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything
import library.feature_extraction as fe


def create_orthogonal_centroids(df_features):
    """
    Transforms the 12-view feature DataFrame into a 3-centroid densified DataFrame.

    Logic:
    1. Sorts data by ID and View Index to ensure alignment.
    2. Uses vectorized operations to reshape features into (N_images, 12, N_features).
    3. Computes averages for Centroids A, B, and C based on Config indices.
    4. Replicates invariant tabular features and metadata.

    Args:
        df_features (pd.DataFrame): DataFrame containing 12 rows per image with visual features.

    Returns:
        pd.DataFrame: Densified DataFrame with 3 rows per image (Centroids A, B, C).
    """
    # Ensure strict ordering for vectorization
    # We expect 12 rows per ID.
    df_sorted = df_features.sort_values(by=["id", "view_idx"]).reset_index(drop=True)

    # Identify columns
    feature_cols = [c for c in df_sorted.columns if c.startswith("feat_")]
    # Tabular columns: margin, shape, texture
    tabular_cols = [
        c
        for c in df_sorted.columns
        if c.startswith("margin") or c.startswith("shape") or c.startswith("texture")
    ]
    # Metadata columns
    meta_cols = ["id"]
    if "species" in df_sorted.columns:
        meta_cols.append("species")

    # Extract numpy arrays
    # Shape: (N_samples, N_features)
    visual_data = df_sorted[feature_cols].values
    num_samples = len(df_sorted)
    num_views = Config.NUM_ROTATIONS
    num_images = num_samples // num_views

    if num_samples % num_views != 0:
        raise ValueError(
            f"Total samples ({num_samples}) is not divisible by number of views ({num_views}). Data integrity error."
        )

    num_visual_feats = visual_data.shape[1]

    # Reshape to (N_images, 12, N_visual_features)
    visual_tensor = visual_data.reshape(num_images, num_views, num_visual_feats)

    # Compute Centroids
    # Indices from Config: {'A': [0,3,6,9], 'B': [1,4,7,10], 'C': [2,5,8,11]}
    indices_a = Config.CENTROID_INDICES["A"]
    indices_b = Config.CENTROID_INDICES["B"]
    indices_c = Config.CENTROID_INDICES["C"]

    # Mean across the view dimension (axis 1) for specific indices
    centroid_a = visual_tensor[:, indices_a, :].mean(axis=1)  # (N_images, N_feats)
    centroid_b = visual_tensor[:, indices_b, :].mean(axis=1)
    centroid_c = visual_tensor[:, indices_c, :].mean(axis=1)

    # Stack centroids: Order -> Image1_A, Image1_B, Image1_C, Image2_A, ...
    # To achieve this, we stack along a new axis and then reshape
    # Stack shape: (N_images, 3, N_feats)
    centroids_stacked = np.stack([centroid_a, centroid_b, centroid_c], axis=1)
    # Flatten to (N_images * 3, N_feats)
    visual_densified = centroids_stacked.reshape(num_images * 3, num_visual_feats)

    # Handle Tabular and Metadata (Invariant per image)
    # We take the first view (index 0) for each image
    # df_sorted is sorted by id, view_idx. Every 12th row starting at 0 is view 0.
    df_invariant = df_sorted.iloc[::num_views].reset_index(drop=True)

    # Replicate invariant data 3 times per image to match centroids
    # We use numpy repeat. index 0 -> 0, 0, 0; index 1 -> 1, 1, 1
    # This matches the order of visual_densified (A, B, C per image)
    invariant_ids = df_invariant["id"].values
    invariant_ids_repeated = np.repeat(invariant_ids, 3)

    # Construct result DataFrame
    # 1. IDs and Centroid Labels
    centroid_labels = np.tile(["A", "B", "C"], num_images)

    df_result = pd.DataFrame(
        {"id": invariant_ids_repeated, "centroid_type": centroid_labels}
    )

    # 2. Add Visual Features
    df_visual = pd.DataFrame(visual_densified, columns=feature_cols)
    df_result = pd.concat([df_result, df_visual], axis=1)

    # 3. Add Tabular Features and Metadata
    # Repeat tabular values
    if tabular_cols:
        tab_data = df_invariant[tabular_cols].values
        tab_data_repeated = np.repeat(tab_data, 3, axis=0)
        df_tab = pd.DataFrame(tab_data_repeated, columns=tabular_cols)
        df_result = pd.concat([df_result, df_tab], axis=1)

    # 4. Add Species if exists
    if "species" in meta_cols:
        species_data = df_invariant["species"].values
        species_repeated = np.repeat(species_data, 3)
        df_result["species"] = species_repeated

    return df_result


def get_processed_data(mode="train", load_cached_data=True, sample_limit=None):
    """
    Main entry point to retrieve densified data.

    Pipeline:
    1. Check for cached densified parquet file.
    2. If missing/force reload:
       a. Get raw 12-view features (via feature_extraction module).
       b. Compute orthogonal centroids.
       c. Cache result.
    3. Return DataFrame.

    Args:
        mode (str): 'train' or 'test'.
        load_cached_data (bool): Whether to use disk cache.
        sample_limit (int): Debug limit.

    Returns:
        pd.DataFrame: The processed densified dataset.
    """
    seed_everything(Config.RANDOM_SEED)

    # Define cache path for the final densified data
    # Note: Config has paths for raw features, we define one for processed here
    filename = f"{mode}_densified.parquet"
    if sample_limit is not None:
        filename = f"{mode}_densified_limit{sample_limit}.parquet"

    cache_path = os.path.join(Config.WORKING_DIR, filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached densified {mode} data from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {mode} data (Densification)...")

    # Get raw 12-view features (this handles its own caching of the raw extraction)
    df_raw = fe.get_or_compute_features(
        mode=mode, load_cached_data=load_cached_data, sample_limit=sample_limit
    )

    if df_raw.empty:
        print("Warning: Raw feature extraction returned empty DataFrame.")
        return pd.DataFrame()

    # Create Centroids
    df_densified = create_orthogonal_centroids(df_raw)

    # 3. Save Cache
    print(f"Saving {len(df_densified)} densified rows to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_densified.to_parquet(cache_path, index=False)

    return df_densified


def get_stratified_folds(df, n_folds=None, seed=None):
    """
    Generates Stratified K-Fold indices based on unique Image IDs.
    Ensures that all 3 centroids of a single image stay in the same fold.

    Args:
        df (pd.DataFrame): The densified DataFrame (must contain 'id' and 'species').
        n_folds (int): Number of folds. Defaults to Config.N_FOLDS.
        seed (int): Random seed. Defaults to Config.RANDOM_SEED.

    Returns:
        list of tuples: [(train_indices, val_indices), ...]
        Indices refer to the rows in the input df.
    """
    if n_folds is None:
        n_folds = Config.N_FOLDS
    if seed is None:
        seed = Config.RANDOM_SEED

    if "species" not in df.columns:
        raise ValueError("Cannot perform stratified split without 'species' column.")

    # 1. Get unique IDs and their labels
    # Since data is densified (repeated), we drop duplicates to get 1 row per image
    df_unique = (
        df[["id", "species"]].drop_duplicates(subset=["id"]).reset_index(drop=True)
    )

    X_unique = df_unique["id"].values
    y_unique = df_unique["species"].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []

    # Map ID to list of dataframe indices
    # This speeds up lookup compared to df[df['id'].isin(ids)].index
    id_to_indices = df.groupby("id").indices  # dict: id -> array of indices

    print(
        f"Generating {n_folds} stratified folds based on {len(df_unique)} unique images..."
    )

    for fold_idx, (train_id_idx, val_id_idx) in enumerate(
        skf.split(X_unique, y_unique)
    ):
        # Get the actual IDs for this split
        train_ids = X_unique[train_id_idx]
        val_ids = X_unique[val_id_idx]

        # Convert IDs to DataFrame indices
        # Concatenate the index arrays for all IDs in the set
        train_indices = np.concatenate([id_to_indices[uid] for uid in train_ids])
        val_indices = np.concatenate([id_to_indices[uid] for uid in val_ids])

        # Shuffle train indices within the fold (optional but good practice)
        np.random.shuffle(train_indices)
        # Sort validation indices for deterministic evaluation order
        val_indices.sort()

        folds.append((train_indices, val_indices))

    return folds
