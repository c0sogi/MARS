import os
import json
import pandas as pd
import numpy as np
from library.config import Config


class NotebookParser:
    """
    Helper class to parse notebook JSON files and extract cell data.
    """

    @staticmethod
    def parse_notebook(filepath, notebook_id, cell_order_str=None, is_train=True):
        """
        Parses a single notebook JSON file.

        Args:
            filepath (str): Path to the JSON file.
            notebook_id (str): ID of the notebook.
            cell_order_str (str, optional): Space-delimited string of correct cell order.
                                            Required if is_train is True.
            is_train (bool): Whether this is a training/validation notebook (has labels).

        Returns:
            tuple: (md_cells_data, code_cells_data)
                md_cells_data (list of dict): List of markdown cell info.
                code_cells_data (dict): Dictionary with 'ids' and 'sources' lists for code cells.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return [], {"ids": [], "sources": []}

        cell_types = data.get("cell_type", {})
        sources = data.get("source", {})

        # Prepare containers
        md_cells = []
        code_ids = []
        code_sources = []

        if is_train:
            if not cell_order_str:
                return [], {"ids": [], "sources": []}

            # Use ground truth order to determine ranks and code sequence
            full_order = cell_order_str.split()
            total_cells = len(full_order)

            for rank, cell_id in enumerate(full_order):
                ctype = cell_types.get(cell_id, "unknown")
                source_text = sources.get(cell_id, "")

                if ctype == "code":
                    code_ids.append(cell_id)
                    code_sources.append(source_text)
                elif ctype == "markdown":
                    # Normalized rank: 0.0 (top) to 1.0 (bottom)
                    norm_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0
                    md_cells.append(
                        {
                            "cell_id": cell_id,
                            "notebook_id": notebook_id,
                            "source": source_text,
                            "rank": norm_rank,
                        }
                    )
        else:
            # For Test set:
            # Code cells are in original (correct) order in the JSON keys (Python 3.7+ dicts preserve insertion order).
            # Markdown cells are shuffled and placed after code cells in the JSON.
            # We iterate through the JSON keys to extract Code cells in order.
            # We extract Markdown cells without rank.

            # Note: The prompt states "The code cells are in their original (correct) order."
            # We rely on the order of keys in the 'source' dictionary.

            for cell_id, source_text in sources.items():
                ctype = cell_types.get(cell_id, "unknown")

                if ctype == "code":
                    code_ids.append(cell_id)
                    code_sources.append(source_text)
                elif ctype == "markdown":
                    md_cells.append(
                        {
                            "cell_id": cell_id,
                            "notebook_id": notebook_id,
                            "source": source_text,
                            "rank": np.nan,  # No target for test
                        }
                    )

        return md_cells, {"ids": code_ids, "sources": code_sources}


def load_dataset(split="train", load_cached_data=True):
    """
    Loads the dataset for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (df_md, df_nb)
            df_md (pd.DataFrame): DataFrame of markdown cells.
            df_nb (pd.DataFrame): DataFrame of notebook contexts (code cells).
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    md_cache_path = os.path.join(cache_dir, f"{split}_md.parquet")
    nb_cache_path = os.path.join(cache_dir, f"{split}_nb.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(md_cache_path) and os.path.exists(nb_cache_path):
            print(f"[{split.upper()}] Loading data from cache...")
            try:
                df_md = pd.read_parquet(md_cache_path, engine="pyarrow")
                df_nb = pd.read_parquet(nb_cache_path, engine="pyarrow")
                print(
                    f"[{split.upper()}] Loaded {len(df_md)} markdown cells and {len(df_nb)} notebooks."
                )
                return df_md, df_nb
            except Exception as e:
                print(f"[{split.upper()}] Cache load failed ({e}). Reprocessing...")

    # 2. Load Metadata
    print(f"[{split.upper()}] Loading metadata...")
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Debugging: Sample if configured
    if Config.DEBUG:
        print(
            f"[{split.upper()}] Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} notebooks."
        )
        if len(df_meta) > Config.DEBUG_SAMPLE_SIZE:
            df_meta = df_meta.sample(
                n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.RANDOM_STATE
            ).reset_index(drop=True)

    # 3. Process Notebooks
    print(f"[{split.upper()}] Parsing {len(df_meta)} notebooks...")

    all_md_cells = []
    all_nb_data = []

    # Pre-fetch ancestor_id mapping if available
    ancestor_map = {}
    if "ancestor_id" in df_meta.columns:
        ancestor_map = dict(zip(df_meta["id"], df_meta["ancestor_id"]))

    count = 0
    for _, row in df_meta.iterrows():
        nb_id = row["id"]
        rel_path = row["filepath"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Get cell order if available (Train/Val)
        cell_order = row["cell_order"] if "cell_order" in row else None
        is_train_mode = split in ["train", "val"]

        # Parse
        md_cells, code_data = NotebookParser.parse_notebook(
            full_path, nb_id, cell_order_str=cell_order, is_train=is_train_mode
        )

        # Add ancestor info to md cells
        ancestor_id = ancestor_map.get(nb_id, nb_id)
        for cell in md_cells:
            cell["ancestor_id"] = ancestor_id

        all_md_cells.extend(md_cells)

        all_nb_data.append(
            {
                "notebook_id": nb_id,
                "code_ids": code_data["ids"],  # List of strings
                "code_sources": code_data["sources"],  # List of strings
            }
        )

        count += 1
        if count % 10000 == 0:
            print(f"[{split.upper()}] Processed {count} notebooks...")

    # 4. Construct DataFrames
    df_md = pd.DataFrame(all_md_cells)
    df_nb = pd.DataFrame(all_nb_data)

    # Ensure correct types
    if not df_md.empty:
        df_md["rank"] = df_md["rank"].astype(float)

    # 5. Save to Cache
    print(f"[{split.upper()}] Saving to cache...")
    try:
        df_md.to_parquet(md_cache_path, index=False, engine="pyarrow")
        df_nb.to_parquet(nb_cache_path, index=False, engine="pyarrow")
    except Exception as e:
        print(f"[{split.upper()}] Warning: Failed to save cache ({e})")

    print(
        f"[{split.upper()}] Processing complete. MD Cells: {len(df_md)}, Notebooks: {len(df_nb)}"
    )
    return df_md, df_nb
