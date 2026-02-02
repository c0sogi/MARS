import os
import json
import pandas as pd
import numpy as np
from library.config import Config


class NotebookLoader:
    """
    Helper class to load and parse notebook JSON files.
    """

    @staticmethod
    def read_json(filepath):
        """
        Reads a JSON file from the given path.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def parse_notebook(notebook_id, json_data, cell_order_str=None):
        """
        Parses a single notebook's JSON data into a list of cell dictionaries.

        Args:
            notebook_id (str): The notebook identifier.
            json_data (dict): The content of the notebook JSON.
            cell_order_str (str, optional): Space-delimited string of cell IDs representing
                                            the ground truth order. Used for train/val sets.

        Returns:
            list: A list of dictionaries, each representing a cell with metadata.
        """
        cell_types = json_data.get("cell_type", {})
        sources = json_data.get("source", {})

        cells = []

        if cell_order_str:
            # Training/Validation mode: Ground truth order is provided.
            order = cell_order_str.split()
            total_cells = len(order)

            for rank, cell_id in enumerate(order):
                c_type = cell_types.get(cell_id, "unknown")
                c_source = sources.get(cell_id, "")

                # Calculate Normalized Rank (Target)
                # Avoid division by zero for single-cell notebooks
                if total_cells > 1:
                    pct_rank = rank / (total_cells - 1)
                else:
                    pct_rank = 0.0

                cells.append(
                    {
                        "id": notebook_id,
                        "cell_id": cell_id,
                        "cell_type": c_type,
                        "source": c_source,
                        "rank": rank,
                        "pct_rank": pct_rank,
                        # ancestor_id will be populated from metadata in the main loop
                    }
                )
        else:
            # Test mode: Ground truth order is unknown.
            # We rely on the fact that code cells are in the correct relative order in the JSON.
            # Markdown cells are shuffled.

            all_cell_ids = list(cell_types.keys())

            # Identify code cells (anchors) and markdown cells
            code_cells = [cid for cid in all_cell_ids if cell_types.get(cid) == "code"]
            markdown_cells = [
                cid for cid in all_cell_ids if cell_types.get(cid) == "markdown"
            ]

            # Process Code Cells
            for i, cell_id in enumerate(code_cells):
                cells.append(
                    {
                        "id": notebook_id,
                        "cell_id": cell_id,
                        "cell_type": "code",
                        "source": sources.get(cell_id, ""),
                        "rank": -1,  # Absolute rank unknown
                        "pct_rank": -1.0,  # Target unknown
                        "code_rank": i,  # Relative rank among code cells (useful for anchors)
                    }
                )

            # Process Markdown Cells
            for cell_id in markdown_cells:
                cells.append(
                    {
                        "id": notebook_id,
                        "cell_id": cell_id,
                        "cell_type": "markdown",
                        "source": sources.get(cell_id, ""),
                        "rank": -1,
                        "pct_rank": -1.0,
                        "code_rank": -1,
                    }
                )

        return cells


def get_partition_data(partition="train", load_cached_data=True, debug=False):
    """
    Retrieves the processed dataframe for a specific data partition.
    Implements caching logic to avoid re-processing raw JSONs.

    Args:
        partition (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug (bool): If True, processes only a small subset of data.

    Returns:
        pd.DataFrame: DataFrame containing cell-level information.
    """

    # 1. Resolve File Paths
    if partition == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.CACHE_TRAIN_PROCESSED
    elif partition == "val":
        metadata_path = Config.VAL_METADATA_PATH
        cache_path = Config.CACHE_VAL_PROCESSED
    elif partition == "test":
        metadata_path = Config.TEST_METADATA_PATH
        cache_path = Config.CACHE_TEST_PROCESSED
    else:
        raise ValueError(
            f"Invalid partition '{partition}'. Must be 'train', 'val', or 'test'."
        )

    # 2. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"[{partition}] Loading processed data from cache: {cache_path}")
        df = pd.read_parquet(cache_path)

        # Apply debug sampling on cached data if requested
        if debug:
            unique_ids = df["id"].unique()
            if len(unique_ids) > Config.DEBUG_SAMPLE_SIZE:
                sample_ids = unique_ids[: Config.DEBUG_SAMPLE_SIZE]
                df = df[df["id"].isin(sample_ids)].copy()
                print(
                    f"[{partition}] Debug mode: Sampled {len(sample_ids)} notebooks from cache."
                )
        return df

    # 3. Process from Scratch
    print(f"[{partition}] Processing raw data from scratch...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Apply debug sampling on metadata before processing
    if debug:
        df_meta = df_meta.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()
        print(f"[{partition}] Debug mode: Processing {len(df_meta)} notebooks.")

    all_cells = []

    # Iterate through notebooks
    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        rel_path = row["filepath"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Get ground truth order if available (train/val)
        cell_order = row["cell_order"] if "cell_order" in row else None

        try:
            # Read and Parse
            json_data = NotebookLoader.read_json(full_path)
            cells = NotebookLoader.parse_notebook(nb_id, json_data, cell_order)

            # Attach Ancestor ID (default to nb_id if missing, e.g., in test)
            ancestor_id = row.get("ancestor_id", nb_id)
            for c in cells:
                c["ancestor_id"] = ancestor_id

            all_cells.extend(cells)

        except Exception as e:
            print(f"Error processing notebook {nb_id}: {e}")
            continue

    df_processed = pd.DataFrame(all_cells)

    # 4. Save to Cache
    # Only save if we are NOT in debug mode (to avoid overwriting full cache with partial data)
    if not debug:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"[{partition}] Saving processed data to cache: {cache_path}")
        df_processed.to_parquet(cache_path, index=False)

    return df_processed
