import os
import json
import pandas as pd
import numpy as np
from library.config import Config


class NotebookLoader:
    """
    Handles loading, parsing, and caching of notebook data.
    Separates content into Markdown (target) and Code (anchor) DataFrames.
    """

    def __init__(self):
        self.config = Config

    def load_data(
        self, split: str = "train", load_cached_data: bool = True, debug_n: int = None
    ):
        """
        Loads notebook data for a specific split.

        Args:
            split: One of 'train', 'val', 'test'.
            load_cached_data: If True, attempts to load pre-processed parquet files.
            debug_n: If provided, limits the number of notebooks processed/returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_markdown, df_code)
        """
        # Define cache paths
        cache_dir = self.config.WORKING_DIR
        md_cache_path = os.path.join(cache_dir, f"{split}_markdown.parquet")
        code_cache_path = os.path.join(cache_dir, f"{split}_code.parquet")

        # 1. Attempt to load from cache
        if load_cached_data:
            if os.path.exists(md_cache_path) and os.path.exists(code_cache_path):
                print(f"Loading {split} data from cache: {md_cache_path}")
                df_md = pd.read_parquet(md_cache_path)
                df_code = pd.read_parquet(code_cache_path)

                if debug_n is not None:
                    # Filter for debugging
                    nb_ids = df_md["notebook_id"].unique()[:debug_n]
                    df_md = df_md[df_md["notebook_id"].isin(nb_ids)].reset_index(
                        drop=True
                    )
                    df_code = df_code[df_code["notebook_id"].isin(nb_ids)].reset_index(
                        drop=True
                    )

                return df_md, df_code
            else:
                print(f"Cache miss for {split}. Processing from raw files...")

        # 2. Determine Metadata Source
        if split == "train":
            meta_path = self.config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = self.config.VAL_METADATA_PATH
        elif split == "test":
            meta_path = self.config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Apply debug limit on metadata if processing from scratch
        if debug_n is not None:
            df_meta = df_meta.iloc[:debug_n]

        # 3. Process Notebooks
        md_list = []
        code_list = []

        # Pre-fetch columns for performance
        ids = df_meta["id"].values
        filepaths = df_meta["filepath"].values

        # 'ancestor_id' and 'cell_order' exist only for train/val
        has_ancestor = "ancestor_id" in df_meta.columns
        ancestors = (
            df_meta["ancestor_id"].values if has_ancestor else [None] * len(df_meta)
        )

        has_order = "cell_order" in df_meta.columns
        orders = df_meta["cell_order"].values if has_order else [None] * len(df_meta)

        input_dir = self.config.INPUT_DIR

        for i in range(len(df_meta)):
            nb_id = ids[i]
            rel_path = filepaths[i]
            ancestor = ancestors[i]
            order_str = orders[i]

            full_path = os.path.join(input_dir, rel_path)

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    nb_json = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read {full_path}. Error: {e}")
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            if has_order and isinstance(order_str, str):
                # --- TRAIN / VAL LOGIC ---
                # We have ground truth order.
                cell_order = order_str.split()
                total_cells = len(cell_order)

                for rank, cell_id in enumerate(cell_order):
                    ctype = cell_types.get(cell_id, "unknown")
                    src = sources.get(cell_id, "")

                    # Calculate normalized rank [0.0, 1.0]
                    norm_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0

                    entry = {
                        "notebook_id": nb_id,
                        "cell_id": cell_id,
                        "source": src,
                        "rank": norm_rank,
                        "ancestor_id": ancestor,
                    }

                    if ctype == "markdown":
                        md_list.append(entry)
                    elif ctype == "code":
                        code_list.append(entry)

            else:
                # --- TEST LOGIC ---
                # No ground truth order.
                # We assume code cells in JSON are in the correct relative order (skeleton).
                # Markdown cells have unknown rank.

                # Iterate over keys (Python 3.7+ preserves insertion order)
                for cell_id, ctype in cell_types.items():
                    src = sources.get(cell_id, "")

                    entry = {
                        "notebook_id": nb_id,
                        "cell_id": cell_id,
                        "source": src,
                        "rank": -1.0,  # Unknown rank
                        "ancestor_id": nb_id,  # Use self ID as ancestor for test
                    }

                    if ctype == "markdown":
                        md_list.append(entry)
                    elif ctype == "code":
                        code_list.append(entry)

        # 4. Construct DataFrames
        df_md = pd.DataFrame(md_list)
        df_code = pd.DataFrame(code_list)

        # Enforce data types to save memory
        if not df_md.empty:
            df_md["rank"] = df_md["rank"].astype(np.float32)
            df_md["source"] = df_md["source"].astype(str)

        if not df_code.empty:
            df_code["rank"] = df_code["rank"].astype(np.float32)
            df_code["source"] = df_code["source"].astype(str)

        # 5. Save to Cache
        # Only save if we processed the full requested set (not just a debug slice of a raw load)
        if debug_n is None:
            print(f"Saving processed {split} data to cache...")
            os.makedirs(cache_dir, exist_ok=True)
            df_md.to_parquet(md_cache_path, index=False)
            df_code.to_parquet(code_cache_path, index=False)

        return df_md, df_code
