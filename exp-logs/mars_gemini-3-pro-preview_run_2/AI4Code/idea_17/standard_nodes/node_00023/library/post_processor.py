import os
import numpy as np
import pandas as pd
from library.config import Config


class RankSorter:
    """
    Responsible for merging predicted markdown ranks with fixed code cell ranks,
    sorting them to reconstruct the notebook, and generating the submission string.
    """

    def __init__(self):
        self.config = Config

    def sort_notebooks(
        self,
        df_md: pd.DataFrame,
        df_code: pd.DataFrame,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Merges markdown and code cells, sorts them by rank, and generates the cell_order string.

        Args:
            df_md: DataFrame containing markdown cells with columns ['notebook_id', 'cell_id', 'pred_rank'].
            df_code: DataFrame containing code cells with columns ['notebook_id', 'cell_id'].
                     Code cells are assumed to be in the correct relative order per notebook.
            load_cached_data: If True, attempts to load the processed submission DataFrame from cache.

        Returns:
            pd.DataFrame: DataFrame with columns ['id', 'cell_order'].
        """
        cache_path = os.path.join(
            self.config.WORKING_DIR, "sorted_submission_df.parquet"
        )

        # 1. Caching Logic
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading sorted submission data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        print("Sorting notebooks and generating cell orders...")

        # 2. Calculate Code Ranks
        # Code cells serve as the skeleton. We assign them equidistant ranks [0.0, 1.0].
        # We assume df_code is already sorted by notebook and insertion order (as loaded by NotebookLoader).

        def calc_rank(g):
            n = len(g)
            # If 0 or 1 cell, rank is 0.0
            if n <= 1:
                return pd.Series([0.0] * n, index=g.index)
            # Linspace from 0 to 1
            return pd.Series(np.arange(n) / (n - 1), index=g.index)

        # Ensure we don't modify the original dataframe in place unexpectedly
        df_code_processed = df_code.copy()

        # Calculate ranks grouped by notebook
        code_ranks = df_code_processed.groupby("notebook_id", group_keys=False)[
            "cell_id"
        ].apply(calc_rank)

        df_code_processed["pred_rank"] = code_ranks.astype(np.float32)

        # 3. Merge DataFrames
        # Select only necessary columns
        cols = ["notebook_id", "cell_id", "pred_rank"]

        if "pred_rank" not in df_md.columns:
            raise ValueError("Markdown DataFrame must contain 'pred_rank' column.")

        df_combined = pd.concat(
            [
                df_md[cols],
                df_code_processed[cols],
            ],
            axis=0,
            ignore_index=True,
        )

        # 4. Sort
        # Primary sort key: notebook_id
        # Secondary sort key: pred_rank (ascending)
        df_sorted = df_combined.sort_values(["notebook_id", "pred_rank"])

        # 5. Aggregate to String
        # Group by notebook and join cell_ids with space
        submission_series = df_sorted.groupby("notebook_id")["cell_id"].apply(
            lambda x: " ".join(x)
        )

        # Convert to DataFrame matching submission format
        submission_df = submission_series.reset_index()
        submission_df.columns = ["id", "cell_order"]

        # 6. Save to Cache
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)
        print(f"Saving sorted submission data to cache: {cache_path}")
        submission_df.to_parquet(cache_path, index=False)

        return submission_df


class SubmissionWriter:
    """
    Handles writing the final submission file to disk.
    """

    def __init__(self):
        self.config = Config

    def save_submission(self, submission_df: pd.DataFrame):
        """
        Saves the submission DataFrame to the path defined in Config.

        Args:
            submission_df: DataFrame with columns ['id', 'cell_order'].
        """
        output_path = self.config.SUBMISSION_PATH
        output_dir = os.path.dirname(output_path)

        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving final submission CSV to {output_path}")
        submission_df.to_csv(output_path, index=False)
