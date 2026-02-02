import os
import numpy as np
import pandas as pd
from library.config import Config
from library.sparse_engine import SparseRanker
from library.dense_engine import DenseEngine
from library.data_factory import load_data_factory


def predict_hybrid(df_test: pd.DataFrame) -> np.ndarray:
    """
    Generates hybrid rank predictions using both Sparse and Dense streams.

    Args:
        df_test (pd.DataFrame): Test dataframe containing markdown cells.

    Returns:
        np.ndarray: Combined predicted ranks.
    """
    print("Loading models for inference...")

    # Load Sparse Stream
    sparse_model = SparseRanker()
    sparse_model.load()
    print("Generating Sparse Stream predictions...")
    pred_sparse = sparse_model.predict(df_test)

    # Load Dense Stream
    dense_model = DenseEngine()
    dense_model.load()
    print("Generating Dense Stream predictions...")
    pred_dense = dense_model.predict(df_test)

    # Weighted Ensemble
    print(f"Combining predictions with Alpha={Config.ALPHA}...")
    pred_final = Config.ALPHA * pred_sparse + (1 - Config.ALPHA) * pred_dense

    return pred_final


def sort_notebooks(
    df_test: pd.DataFrame, predictions: np.ndarray, test_anchors: dict
) -> pd.DataFrame:
    """
    Reconstructs the cell order for each notebook by combining fixed code anchors
    and predicted markdown ranks.

    Args:
        df_test (pd.DataFrame): Test dataframe containing 'id' and 'cell_id'.
        predictions (np.ndarray): Predicted ranks for the markdown cells in df_test.
        test_anchors (dict): Mapping of notebook_id -> list of code cell IDs.

    Returns:
        pd.DataFrame: Submission dataframe with 'id' and 'cell_order'.
    """
    print("Sorting notebooks based on hybrid predictions and anchors...")

    # Attach predictions to the dataframe
    df_test = df_test.copy()
    df_test["pred_rank"] = predictions

    # Pre-group markdown cells by notebook ID for faster access
    # Result: {nb_id: [(cell_id, rank), ...]}
    md_groups = df_test.groupby("id")
    md_map = {}
    for nb_id, group in md_groups:
        md_map[nb_id] = list(zip(group["cell_id"].values, group["pred_rank"].values))

    submission_rows = []

    # Iterate over all notebooks in the test set (keys from anchors)
    for nb_id, code_cells in test_anchors.items():
        # 1. Assign ranks to Code Cells (Anchors)
        # We distribute them equidistantly from 0.0 to 1.0 to form the skeleton.
        n_code = len(code_cells)
        if n_code == 0:
            code_ranked = []
        else:
            # If there is only 1 code cell, we place it at 0.0 (top)
            # If multiple, we span 0.0 to 1.0
            if n_code == 1:
                ranks = [0.0]
            else:
                ranks = np.linspace(0.0, 1.0, n_code)
            code_ranked = list(zip(code_cells, ranks))

        # 2. Retrieve predicted Markdown Cells
        md_ranked = md_map.get(nb_id, [])

        # 3. Combine and Sort
        # We combine both lists. The sort key is the rank.
        # Stability note: If ranks are equal, Python's stable sort preserves order.
        # Putting code_ranked first means code cells win ties against markdown cells predicted at exact same rank.
        all_cells = code_ranked + md_ranked
        all_cells.sort(key=lambda x: x[1])

        # 4. Extract ordered IDs
        cell_order_str = " ".join([x[0] for x in all_cells])
        submission_rows.append({"id": nb_id, "cell_order": cell_order_str})

    return pd.DataFrame(submission_rows)


def generate_submission(load_cached_data: bool = True):
    """
    Main driver function to generate the submission file.

    Args:
        load_cached_data (bool): Whether to use cached pre-processed data.
    """
    print("\n" + "=" * 40)
    print("Starting Inference and Submission Generation")
    print("=" * 40)

    # 1. Load Test Data
    # We ignore train/val returns here
    _, _, df_test, test_anchors = load_data_factory(load_cached_data=load_cached_data)

    # 2. Generate Predictions
    if len(df_test) > 0:
        predictions = predict_hybrid(df_test)
    else:
        # Handle edge case where test set might have no markdown cells (unlikely but possible)
        predictions = np.array([])

    # 3. Sort and Format
    df_submission = sort_notebooks(df_test, predictions, test_anchors)

    # 4. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission generated successfully with {len(df_submission)} rows.")
    print(f"Saved to: {Config.SUBMISSION_PATH}")
