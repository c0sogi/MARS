import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import read_notebook_json, load_or_save_cache


class DataLoader:
    """
    Handles loading and processing of notebook data for the AI4Code task.
    Implements caching and data extraction logic for both markdown (targets) and code (context) cells.
    """

    @staticmethod
    def load_data(
        split="train", load_cached_data=True, debug=False, num_debug_samples=100
    ):
        """
        Loads the dataset for a specific split (train, val, test).
        Returns two DataFrames: one for markdown cells (targets) and one for code cells (context).

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.
            debug (bool): If True, processes only a small subset of data.
            num_debug_samples (int): Number of notebooks to process in debug mode.

        Returns:
            tuple: (df_markdown, df_code)
                - df_markdown: DataFrame containing markdown cells, ranks, and metadata.
                - df_code: DataFrame containing code cells and their ranks.
        """
        # Determine paths and cache names based on split
        if split == "train":
            metadata_path = Config.TRAIN_METADATA_PATH
            # Use the config path for markdown, derive code path
            cache_name_md = os.path.basename(Config.TRAIN_CACHE_PATH)
            cache_name_code = cache_name_md.replace(".parquet", "_code.parquet")
        elif split == "val":
            metadata_path = Config.VAL_METADATA_PATH
            cache_name_md = os.path.basename(Config.VAL_CACHE_PATH)
            cache_name_code = cache_name_md.replace(".parquet", "_code.parquet")
        elif split == "test":
            metadata_path = Config.TEST_METADATA_PATH
            cache_name_md = os.path.basename(Config.TEST_CACHE_PATH)
            cache_name_code = cache_name_md.replace(".parquet", "_code.parquet")
        else:
            raise ValueError(f"Invalid split: {split}")

        # Modify cache names for debug mode to prevent polluting main cache
        if debug:
            cache_name_md = f"debug_{cache_name_md}"
            cache_name_code = f"debug_{cache_name_code}"

        # Define producer functions for the caching utility
        # Note: _process_notebooks returns (df_md, df_code). We wrap it to return just one.
        def _produce_markdown_data():
            df_md, _ = DataLoader._process_notebooks(
                metadata_path, split, debug, num_debug_samples
            )
            return df_md

        def _produce_code_data():
            _, df_code = DataLoader._process_notebooks(
                metadata_path, split, debug, num_debug_samples
            )
            return df_code

        print(f"Loading {split} data (Markdown)...")
        df_markdown = load_or_save_cache(
            file_name=cache_name_md,
            data_producer_fn=_produce_markdown_data,
            load_cached_data=load_cached_data,
        )

        print(f"Loading {split} data (Code)...")
        df_code = load_or_save_cache(
            file_name=cache_name_code,
            data_producer_fn=_produce_code_data,
            load_cached_data=load_cached_data,
        )

        return df_markdown, df_code

    @staticmethod
    def _process_notebooks(metadata_path, split, debug, num_debug_samples):
        """
        Internal function to read metadata and JSONs, returning processed DataFrames.
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        if debug:
            df_meta = df_meta.head(num_debug_samples).copy()

        markdown_rows = []
        code_rows = []

        # Iterate over notebooks
        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            filepath = row["filepath"]

            # Load JSON content
            nb_json = read_notebook_json(filepath)
            if not nb_json:
                continue

            cell_types = nb_json.get("cell_type", {})
            sources = nb_json.get("source", {})

            # Determine Cell Order
            if split in ["train", "val"]:
                # For training/validation, use the ground truth order from metadata
                if "cell_order" not in row or pd.isna(row["cell_order"]):
                    continue
                cell_order = row["cell_order"].split()
            else:
                # For test, we rely on the order in the JSON keys.
                # In this dataset, test JSONs have code cells in correct relative order,
                # and markdown cells appended or shuffled.
                cell_order = list(cell_types.keys())

            # Process Cells
            if split in ["train", "val"]:
                total_cells = len(cell_order)
                for rank, cell_id in enumerate(cell_order):
                    c_type = cell_types.get(cell_id, "unknown")
                    c_source = sources.get(cell_id, "")

                    if c_type == "markdown":
                        # Calculate Normalized Rank: rank / (N-1)
                        norm_rank = rank / (total_cells - 1) if total_cells > 1 else 0.0
                        markdown_rows.append(
                            {
                                "notebook_id": nb_id,
                                "cell_id": cell_id,
                                "source": c_source,
                                "rank": rank,
                                "norm_rank": norm_rank,
                            }
                        )
                    elif c_type == "code":
                        code_rows.append(
                            {
                                "notebook_id": nb_id,
                                "cell_id": cell_id,
                                "source": c_source,
                                "rank": rank,
                            }
                        )
            else:
                # Test Split Logic
                # Code cells: Assign sequential ranks (0, 1, 2...) based on appearance.
                # Markdown cells: Assign placeholder rank (-1).
                current_code_rank = 0
                for cell_id in cell_order:
                    c_type = cell_types.get(cell_id, "unknown")
                    c_source = sources.get(cell_id, "")

                    if c_type == "code":
                        code_rows.append(
                            {
                                "notebook_id": nb_id,
                                "cell_id": cell_id,
                                "source": c_source,
                                "rank": current_code_rank,
                            }
                        )
                        current_code_rank += 1
                    elif c_type == "markdown":
                        markdown_rows.append(
                            {
                                "notebook_id": nb_id,
                                "cell_id": cell_id,
                                "source": c_source,
                                "rank": -1,
                                "norm_rank": -1.0,
                            }
                        )

        # Create DataFrames
        df_markdown = pd.DataFrame(markdown_rows)
        df_code = pd.DataFrame(code_rows)

        # Ensure correct data types
        if not df_markdown.empty:
            df_markdown["rank"] = df_markdown["rank"].astype(int)
            df_markdown["norm_rank"] = df_markdown["norm_rank"].astype(float)

        if not df_code.empty:
            df_code["rank"] = df_code["rank"].astype(int)

        return df_markdown, df_code
