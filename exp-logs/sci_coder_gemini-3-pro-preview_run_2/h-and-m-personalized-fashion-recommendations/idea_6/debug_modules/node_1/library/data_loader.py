import pandas as pd
import numpy as np
import torch
import os
from typing import Tuple, Dict, List
from library.config import Paths, DATA_CONFIG
from library.utils import reduce_mem_usage, CacheManager, setup_logger

# Initialize logger
logger = setup_logger("data_loader")


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the raw metadata parquets for train, val, and test.
    Converts t_dat to datetime and optimizes memory usage.

    Returns:
        train_df: Historical transactions for training customers.
        val_df: Historical transactions for validation customers.
        test_df: Submission template with customers to predict.
    """
    logger.info("Loading raw data from metadata...")

    train_path = Paths.METADATA_DIR / "train.parquet"
    val_path = Paths.METADATA_DIR / "val.parquet"
    test_path = Paths.METADATA_DIR / "test.parquet"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation script has run."
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # Convert dates
    train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])
    val_df["t_dat"] = pd.to_datetime(val_df["t_dat"])

    # Optimize memory
    train_df = reduce_mem_usage(train_df, verbose=False)
    val_df = reduce_mem_usage(val_df, verbose=False)

    logger.info(
        f"Loaded Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )
    return train_df, val_df, test_df


def create_time_split(
    df: pd.DataFrame, val_days: int = DATA_CONFIG["val_days"]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a transaction DataFrame into training and validation sets based on the last `val_days`.
    This simulates the test scenario where we predict the next `val_days` based on history.

    Args:
        df: Input dataframe with 't_dat' column.
        val_days: Number of days to include in the validation set.

    Returns:
        train_split: Data before the cutoff date.
        val_split: Data after the cutoff date (last `val_days`).
    """
    logger.info(f"Creating time split (Last {val_days} days as validation)...")

    max_date = df["t_dat"].max()
    split_date = max_date - pd.Timedelta(days=val_days)

    train_split = df[df["t_dat"] <= split_date].copy()
    val_split = df[df["t_dat"] > split_date].copy()

    logger.info(f"Split Date: {split_date}")
    logger.info(f"Train Split: {train_split.shape}, Val Split: {val_split.shape}")

    return train_split, val_split


def prepare_graph_data(
    df: pd.DataFrame, load_cached: bool = True
) -> Tuple[torch.LongTensor, Dict[str, int], Dict[str, int]]:
    """
    Prepares data for LightGCN by constructing a User-Item bipartite graph.

    1. Filters data based on DATA_CONFIG['graph_window_days'].
    2. Maps customer_id and article_id to contiguous integers.
    3. Constructs the edge_index tensor (2, num_edges).

    Args:
        df: The training transactions dataframe.
        load_cached: Whether to load from cache if available.

    Returns:
        edge_index: torch.LongTensor of shape [2, num_edges]. Row 0 is User IDs, Row 1 is Item IDs.
        user_map: Dictionary mapping customer_id (str) to integer index.
        item_map: Dictionary mapping article_id (str) to integer index.
    """
    cache = CacheManager()

    # Define cache filenames
    edge_file = "graph_edge_index.pt"
    user_map_file = "graph_user_map.npy"
    item_map_file = "graph_item_map.npy"

    # Check cache
    if load_cached:
        if (
            cache.exists(edge_file)
            and cache.exists(user_map_file)
            and cache.exists(item_map_file)
        ):
            logger.info("Loading graph data from cache...")

            # Load tensor
            edge_index = torch.load(cache.get_path(edge_file))

            # Load maps (saved as numpy arrays of keys, where index is the ID)
            user_keys = cache.load_npy(user_map_file)
            item_keys = cache.load_npy(item_map_file)

            user_map = {k: i for i, k in enumerate(user_keys)}
            item_map = {k: i for i, k in enumerate(item_keys)}

            logger.info(
                f"Graph loaded. Users: {len(user_map)}, Items: {len(item_map)}, Edges: {edge_index.shape[1]}"
            )
            return edge_index, user_map, item_map

    logger.info("Processing graph data from scratch...")

    # 1. Filter by time window
    window_days = DATA_CONFIG["graph_window_days"]
    max_date = df["t_dat"].max()
    cutoff_date = max_date - pd.Timedelta(days=window_days)

    logger.info(
        f"Filtering graph data to last {window_days} days (>= {cutoff_date})..."
    )
    filtered_df = df[df["t_dat"] >= cutoff_date].copy()

    # 2. Create Mappings
    # Get unique users and items in the filtered window
    unique_users = filtered_df["customer_id"].unique()
    unique_items = filtered_df["article_id"].unique()

    # Create maps
    user_map = {u: i for i, u in enumerate(unique_users)}
    item_map = {i: k for k, i in enumerate(unique_items)}

    # 3. Create Edge Index
    logger.info("Mapping IDs to integers...")

    # Map IDs in dataframe to integers
    user_indices = filtered_df["customer_id"].map(user_map).astype(np.int64)
    item_indices = filtered_df["article_id"].map(item_map).astype(np.int64)

    # Stack to create [2, num_edges]
    # Row 0: User Indices, Row 1: Item Indices
    edge_index_np = np.vstack((user_indices.values, item_indices.values))
    edge_index = torch.from_numpy(edge_index_np).long()

    # 4. Save to Cache
    logger.info("Saving graph data to cache...")
    torch.save(edge_index, cache.get_path(edge_file))
    cache.save_npy(unique_users, user_map_file)
    cache.save_npy(unique_items, item_map_file)

    logger.info(
        f"Graph processed. Users: {len(user_map)}, Items: {len(item_map)}, Edges: {edge_index.shape[1]}"
    )

    return edge_index, user_map, item_map


def get_recent_popular_items(
    df: pd.DataFrame, top_k: int = 12, days: int = 7
) -> List[str]:
    """
    Get the most popular items from the last `days`.
    Used as a fallback strategy for cold-start users or to fill predictions.

    Args:
        df: Transaction dataframe.
        top_k: Number of items to retrieve.
        days: Time window in days.

    Returns:
        List of popular article_ids.
    """
    max_date = df["t_dat"].max()
    start_date = max_date - pd.Timedelta(days=days)
    recent_df = df[df["t_dat"] > start_date]

    popular = recent_df["article_id"].value_counts().head(top_k).index.tolist()
    return popular
