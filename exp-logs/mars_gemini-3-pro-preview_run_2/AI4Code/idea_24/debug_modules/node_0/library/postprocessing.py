import numpy as np
import pandas as pd
from library.config import SUBMISSION_DIR


class OrderReconstructor:
    """
    Handles the reconstruction of the final cell order from predicted ranks.
    Merges predicted markdown ranks with fixed code cell ranks and sorts them.
    """

    def __init__(self):
        pass

    def reconstruct_order(
        self, test_df: pd.DataFrame, predictions_df: pd.DataFrame
    ) -> dict:
        """
        Reconstructs the cell order for each notebook in the test set.

        Args:
            test_df: DataFrame containing test notebook structure
                     (columns: 'notebook_id', 'cell_id', 'cell_type', ...).
                     Crucially, the code cells in this DF must be in their correct relative order.
            predictions_df: DataFrame containing predictions for markdown cells
                            (columns: 'cell_id', 'pred_rank').

        Returns:
            dict: A dictionary mapping 'notebook_id' to the space-delimited 'cell_order' string.
        """
        # Create a working copy to avoid side effects
        df = test_df.copy()

        # Merge the predicted ranks for markdown cells
        # We use a left join on test_df to ensure we have all cells (code + markdown)
        if "pred_rank" in df.columns:
            df = df.drop(columns=["pred_rank"])

        df = df.merge(
            predictions_df[["cell_id", "pred_rank"]], on="cell_id", how="left"
        )

        predictions = {}

        # Group by notebook to process each independently
        # Note: groupby preserves the order of rows within each group,
        # which is critical for maintaining the relative order of code cells.
        grouped = df.groupby("notebook_id")

        for nb_id, group in grouped:
            # Separate Code and Markdown cells
            is_code = group["cell_type"] != "markdown"

            # --- Handle Code Cells ---
            code_cells = group[is_code].copy()
            n_code = len(code_cells)

            if n_code > 0:
                # Assign fixed, equidistant ranks to code cells from 0.0 to 1.0
                # This acts as the "skeleton" or "anchors" for the notebook
                # np.linspace(0, 1, 1) returns [0.0], which is consistent
                code_cells["pred_rank"] = np.linspace(0.0, 1.0, n_code)

            # --- Handle Markdown Cells ---
            md_cells = group[~is_code].copy()

            # Fill missing predictions with a neutral rank (0.5) for robustness
            # This handles cases where a markdown cell might have been missed in prediction
            if md_cells["pred_rank"].isnull().any():
                md_cells["pred_rank"] = md_cells["pred_rank"].fillna(0.5)

            # --- Combine and Sort ---
            # Concatenate code and markdown cells
            combined = pd.concat([code_cells, md_cells])

            # Sort by the rank (ascending)
            # This interleaves the markdown cells into the fixed code skeleton
            combined = combined.sort_values("pred_rank")

            # --- Format Output ---
            # Extract the cell_ids in the sorted order
            cell_order_list = combined["cell_id"].values

            # Join into a space-delimited string
            cell_order_str = " ".join(cell_order_list)

            predictions[nb_id] = cell_order_str

        return predictions
